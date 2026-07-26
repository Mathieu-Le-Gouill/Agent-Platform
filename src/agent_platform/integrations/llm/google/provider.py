from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from google import genai
from google.genai import types

from agent_platform.core.credentials import resolve_credentials
from agent_platform.core.errors import require_secret
from agent_platform.core.interfaces.llm.response import (
    FinishReason,
    LLMResponse,
    ResponseFormat,
    StreamChunk,
    ToolCallDelta,
)
from agent_platform.core.schemas import model_schema
from agent_platform.core.schemas.message import (
    AssistantMessage,
    AudioBlock,
    ContentBlock,
    ContentMessage,
    ImageBlock,
    Prompt,
    SystemMessage,
    TextBlock,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from agent_platform.core.schemas.token import TokenUsage
from agent_platform.core.tracing import record_token_usage
from agent_platform.integrations.credentials import GoogleCredentials
from agent_platform.integrations.llm._base import NativeLLMProvider
from agent_platform.integrations.llm.google.config import GoogleGenerationConfig

if TYPE_CHECKING:
    from agent_platform.agents.tools.base import Tool


class GoogleLLM(
    NativeLLMProvider[
        GoogleGenerationConfig,
        genai.Client,
        genai.Client,
        types.GenerateContentResponse,
    ]
):
    _provider_name = "google"
    _missing_api_key_message = "Google API key is required but was not provided"

    def __init__(self, credentials: GoogleCredentials | None = None) -> None:
        self._credentials = resolve_credentials(credentials, GoogleCredentials)

    def _tool_to_schema(self, tool: Tool) -> types.Tool:
        return types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters_json_schema=model_schema(tool.input_schema),
                )
            ]
        )

    def _default_config(self) -> GoogleGenerationConfig:
        return GoogleGenerationConfig()

    def _client(self, config: GoogleGenerationConfig) -> genai.Client:
        api_key = require_secret(
            self._credentials.api_key, self._missing_api_key_message
        )
        return genai.Client(api_key=api_key.get_secret_value())

    def _async_client(self, config: GoogleGenerationConfig) -> genai.Client:
        # Single client class, no sync/async split (like Mistral): async calls
        # go through `client.aio.models.*`, sync through `client.models.*`.
        return self._client(config)

    def _sync_client(self, config: GoogleGenerationConfig) -> genai.Client:
        return self._client(config)

    def _invoke_sync(
        self,
        client: genai.Client,
        prompt: Prompt,
        config: GoogleGenerationConfig,
        tools: list[Tool] | None,
    ) -> types.GenerateContentResponse:
        system, contents = _to_native_contents(prompt)
        native_config = _to_native_config(config, system, tools, self._tool_to_schema)
        return client.models.generate_content(
            model=config.model,
            contents=contents,  # type: ignore[arg-type]
            config=native_config,
        )

    async def _invoke_async(
        self,
        client: genai.Client,
        prompt: Prompt,
        config: GoogleGenerationConfig,
        tools: list[Tool] | None,
    ) -> types.GenerateContentResponse:
        system, contents = _to_native_contents(prompt)
        native_config = _to_native_config(config, system, tools, self._tool_to_schema)
        return await client.aio.models.generate_content(
            model=config.model,
            contents=contents,  # type: ignore[arg-type]
            config=native_config,
        )

    def _from_native(
        self, response: types.GenerateContentResponse, model: str
    ) -> LLMResponse:
        return _from_native_response(response, model)

    async def stream(
        self,
        prompt: Prompt,
        config: GoogleGenerationConfig | None = None,
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        config = config or self._default_config()
        client = self._async_client(config)
        system, contents = _to_native_contents(prompt)
        native_config = _to_native_config(config, system, tools, self._tool_to_schema)

        with self._span(config) as span:
            usage_totals = TokenUsage.zero()
            events = await client.aio.models.generate_content_stream(
                model=config.model,
                contents=contents,  # type: ignore[arg-type]
                config=native_config,
            )
            async for chunk in events:
                if chunk.candidates:
                    content = chunk.candidates[0].content
                    parts = content.parts if content else None
                    for idx, part in enumerate(parts or []):
                        if part.text:
                            yield StreamChunk(delta=part.text)
                        if part.function_call:
                            fc = part.function_call
                            # Gemini emits each function call whole in one
                            # chunk (no argument fragmentation like OpenAI/
                            # Anthropic), and has no native call id.
                            yield StreamChunk(
                                delta="",
                                tool_call_deltas=[
                                    ToolCallDelta(
                                        index=idx,
                                        id=fc.id or uuid4().hex,
                                        name=fc.name,
                                        arguments_delta=json.dumps(dict(fc.args or {})),
                                    )
                                ],
                            )
                if chunk.usage_metadata is not None:
                    chunk_usage = TokenUsage(
                        input_tokens=chunk.usage_metadata.prompt_token_count or 0,
                        output_tokens=chunk.usage_metadata.candidates_token_count or 0,
                    )
                    usage_totals = chunk_usage
                    yield StreamChunk(
                        delta="",
                        finish_reason=FinishReason.STOP,
                        usage=chunk_usage,
                    )

            record_token_usage(span, usage_totals)
            yield StreamChunk(delta="", finish_reason=FinishReason.STOP)


# --- Mappers ---


def _block_to_native_part(block: ContentBlock) -> types.Part:
    match block:
        case TextBlock():
            return types.Part.from_text(text=block.text)
        case ImageBlock():
            if isinstance(block.image, str):
                return types.Part.from_uri(file_uri=block.image, mime_type="image/png")
            mime = (
                f"image/{block.image.format.value}"
                if block.image.format
                else "image/png"
            )
            return types.Part.from_bytes(data=block.image.content, mime_type=mime)
        case AudioBlock():
            mime = block.audio.format.value if block.audio.format else "wav"
            return types.Part.from_bytes(
                data=block.audio.content, mime_type=f"audio/{mime}"
            )


def _content_to_native_parts(m: ContentMessage) -> list[types.Part]:
    if isinstance(m.content, str):
        return [types.Part.from_text(text=m.content)] if m.content else []
    return [_block_to_native_part(b) for b in m.blocks]


def _to_native_contents(prompt: Prompt) -> tuple[str | None, list[types.Content]]:
    system_parts: list[str] = []
    contents: list[types.Content] = []

    for m in prompt.messages:
        match m:
            case SystemMessage():
                if m.text:
                    system_parts.append(m.text)
            case UserMessage():
                contents.append(
                    types.Content(role="user", parts=_content_to_native_parts(m))
                )
            case AssistantMessage():
                parts = _content_to_native_parts(m)
                for tc in m.tool_calls:
                    parts.append(
                        types.Part.from_function_call(name=tc.name, args=tc.arguments)
                    )
                contents.append(types.Content(role="model", parts=parts))
            case ToolMessage():
                # Gemini folds tool results into a user-role turn, matched by
                # function name rather than a call id (it has none).
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=m.result.name,
                                response={"content": m.result.content},
                            )
                        ],
                    )
                )

    return ("\n\n".join(system_parts) if system_parts else None, contents)


def _to_native_config(
    config: GoogleGenerationConfig,
    system: str | None,
    tools: list[Tool] | None,
    tool_to_schema: Any,
) -> types.GenerateContentConfig:
    kwargs: dict[str, Any] = {"temperature": config.temperature}
    if config.max_tokens is not None:
        kwargs["max_output_tokens"] = config.max_tokens
    if config.top_p is not None:
        kwargs["top_p"] = config.top_p
    if config.top_k is not None:
        kwargs["top_k"] = config.top_k
    if config.stop_sequences:
        kwargs["stop_sequences"] = config.stop_sequences
    if config.seed is not None:
        kwargs["seed"] = config.seed
    if config.presence_penalty is not None:
        kwargs["presence_penalty"] = config.presence_penalty
    if config.frequency_penalty is not None:
        kwargs["frequency_penalty"] = config.frequency_penalty
    if system:
        kwargs["system_instruction"] = system
    if tools:
        kwargs["tools"] = [tool_to_schema(t) for t in tools]
    if config.thinking_budget is not None:
        kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=config.thinking_budget,
            include_thoughts=config.include_thoughts,
        )

    match config.response_format:
        case ResponseFormat.TEXT:
            pass
        case ResponseFormat.JSON:
            kwargs["response_mime_type"] = "application/json"
        case ResponseFormat.JSON_SCHEMA:
            if config.json_schema is None:
                raise ValueError(
                    "json_schema is required for JSON_SCHEMA response format"
                )
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_json_schema"] = config.json_schema

    kwargs.update(config.extra_params)
    return types.GenerateContentConfig(**kwargs)


def _from_native_response(
    response: types.GenerateContentResponse, model: str
) -> LLMResponse:
    candidate = response.candidates[0] if response.candidates else None
    content = candidate.content if candidate else None
    parts = content.parts if content else []

    text_parts = [part.text for part in (parts or []) if part.text]
    tool_calls = [
        ToolCall(
            id=part.function_call.id or uuid4().hex,
            name=part.function_call.name or "",
            arguments=dict(part.function_call.args or {}),
        )
        for part in (parts or [])
        if part.function_call
    ]

    usage = response.usage_metadata

    return LLMResponse(
        message=AssistantMessage(content="".join(text_parts), tool_calls=tool_calls),
        usage=TokenUsage(
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            output_tokens=(usage.candidates_token_count or 0) if usage else 0,
        ),
        model=model,
        finish_reason=FinishReason.STOP,
    )
