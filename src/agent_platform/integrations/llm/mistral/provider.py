from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from mistralai import Mistral
from mistralai.models import ChatCompletionResponse

from agent_platform.core.credentials import resolve_credentials, resolve_timeout
from agent_platform.core.errors import require_secret
from agent_platform.core.interfaces.llm.response import (
    FinishReason,
    LLMResponse,
    ResponseFormat,
    StreamChunk,
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
from agent_platform.integrations.credentials import MistralCredentials
from agent_platform.integrations.llm._base import NativeLLMProvider
from agent_platform.integrations.llm.mistral.config import MistralGenerationConfig

if TYPE_CHECKING:
    from agent_platform.agents.tools.base import Tool


class MistralLLM(
    NativeLLMProvider[MistralGenerationConfig, Mistral, Mistral, ChatCompletionResponse]
):
    _provider_name = "mistral"

    def __init__(self, credentials: MistralCredentials | None = None) -> None:
        self._credentials = resolve_credentials(credentials, MistralCredentials)

    def _tool_to_schema(self, tool: Tool) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": model_schema(tool.input_schema),
            },
        }

    def _default_config(self) -> MistralGenerationConfig:
        return MistralGenerationConfig()

    def _async_client(self, config: MistralGenerationConfig) -> Mistral:
        api_key = require_secret(
            self._credentials.api_key,
            "Mistral API key is required but was not provided",
        )
        kwargs: dict[str, Any] = {"api_key": api_key.get_secret_value()}
        if self._credentials.base_url:
            kwargs["server_url"] = self._credentials.base_url

        timeout = resolve_timeout(config.timeout, self._credentials)
        if timeout is not None:
            kwargs["timeout_ms"] = int(timeout * 1000)

        # `config.max_retries`/`credentials.max_retries` are intentionally not wired
        # into the native SDK's `retry_config`: unlike OpenAI/Anthropic's simple
        # int attempt count, Mistral's RetryConfig is a time-based backoff
        # (initial_interval/max_interval/exponent/max_elapsed_time) with no
        # attempt-count knob, so there's no faithful translation. The platform's
        # own `@with_retry()` decorator on `agenerate` already provides equivalent
        # attempt-count-based retry behavior.
        return Mistral(**kwargs)

    def _invoke_sync(
        self,
        client: Mistral,
        prompt: Prompt,
        config: MistralGenerationConfig,
        tools: list[Tool] | None,
    ) -> ChatCompletionResponse:
        messages = _to_native_messages(prompt)
        params = _to_native_params(config)
        if tools:
            params["tools"] = [self._tool_to_schema(t) for t in tools]

        return client.chat.complete(
            model=config.model,
            messages=messages,  # type: ignore[arg-type]
            **params,
        )

    async def _invoke_async(
        self,
        client: Mistral,
        prompt: Prompt,
        config: MistralGenerationConfig,
        tools: list[Tool] | None,
    ) -> ChatCompletionResponse:
        messages = _to_native_messages(prompt)
        params = _to_native_params(config)
        if tools:
            params["tools"] = [self._tool_to_schema(t) for t in tools]

        return await client.chat.complete_async(
            model=config.model,
            messages=messages,  # type: ignore[arg-type]
            **params,
        )

    def _from_native(self, response: ChatCompletionResponse, model: str) -> LLMResponse:
        return _from_native_response(response, model)

    async def stream(
        self,
        prompt: Prompt,
        config: MistralGenerationConfig | None = None,
    ) -> AsyncIterator[StreamChunk]:
        config = config or self._default_config()
        client = self._async_client(config)
        messages = _to_native_messages(prompt)
        params = _to_native_params(config)

        with self._span(config) as span:
            usage_totals = TokenUsage.zero()
            events = await client.chat.stream_async(
                model=config.model,
                messages=messages,  # type: ignore[arg-type]
                **params,
            )
            async for event in events:
                chunk = event.data
                if chunk.choices:
                    delta = chunk.choices[0].delta.content
                    if isinstance(delta, str) and delta:
                        yield StreamChunk(delta=delta)
                if chunk.usage is not None:
                    chunk_usage = TokenUsage(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
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


def _block_to_native(block: ContentBlock) -> dict[str, Any]:
    match block:
        case TextBlock():
            return {"type": "text", "text": block.text}
        case ImageBlock():
            if isinstance(block.image, str):
                url = block.image
            else:
                mime = (
                    f"image/{block.image.format.value}"
                    if block.image.format
                    else "image/png"
                )
                data = base64.b64encode(block.image.content).decode()
                url = f"data:{mime};base64,{data}"
            return {"type": "image_url", "image_url": url}
        case AudioBlock():
            data = base64.b64encode(block.audio.content).decode()
            return {"type": "audio", "input_audio": data}


def _content_to_native(m: ContentMessage) -> str | list[dict[str, Any]]:
    if isinstance(m.content, str):
        return m.content
    return [_block_to_native(b) for b in m.blocks]


def _to_native_messages(prompt: Prompt) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for m in prompt.messages:
        match m:
            case SystemMessage():
                result.append({"role": "system", "content": _content_to_native(m)})
            case UserMessage():
                result.append({"role": "user", "content": _content_to_native(m)})
            case AssistantMessage():
                entry: dict[str, Any] = {
                    "role": "assistant",
                    "content": _content_to_native(m),
                }
                if m.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in m.tool_calls
                    ]
                result.append(entry)
            case ToolMessage():
                result.append(
                    {
                        "role": "tool",
                        "content": m.result.content,
                        "tool_call_id": m.result.tool_call_id,
                        "name": m.result.name,
                    }
                )
    return result


def _to_native_params(config: MistralGenerationConfig) -> dict[str, Any]:
    params: dict[str, Any] = {"temperature": config.temperature}
    if config.max_tokens is not None:
        params["max_tokens"] = config.max_tokens
    if config.top_p is not None:
        params["top_p"] = config.top_p
    if config.stop_sequences:
        params["stop"] = config.stop_sequences
    if config.seed is not None:
        params["random_seed"] = config.seed

    if config.frequency_penalty is not None:
        params["frequency_penalty"] = config.frequency_penalty
    if config.presence_penalty is not None:
        params["presence_penalty"] = config.presence_penalty

    match config.response_format:
        case ResponseFormat.JSON:
            params["response_format"] = {"type": "json_object"}
        case ResponseFormat.JSON_SCHEMA:
            if config.json_schema is None:
                raise ValueError(
                    "json_schema is required for JSON_SCHEMA response format"
                )
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "schema": config.json_schema,
                    "name": config.json_schema_name,
                    "strict": config.json_schema_strict,
                },
            }

    params.update(config.extra_params)
    return params


def _from_native_response(response: ChatCompletionResponse, model: str) -> LLMResponse:
    message = response.choices[0].message

    tool_calls = [
        ToolCall(
            # Mistral's SDK types `tc.id` as optional, but the API always assigns
            # one; the uuid4 fallback only guards against that type/reality gap.
            id=tc.id or uuid4().hex,
            name=tc.function.name,
            arguments=(
                tc.function.arguments
                if isinstance(tc.function.arguments, dict)
                else json.loads(tc.function.arguments)
            ),
        )
        for tc in (message.tool_calls or [])
    ]

    content = message.content if isinstance(message.content, str) else ""
    usage = response.usage

    return LLMResponse(
        message=AssistantMessage(content=content or "", tool_calls=tool_calls),
        usage=TokenUsage(
            input_tokens=(usage.prompt_tokens or 0) if usage else 0,
            output_tokens=(usage.completion_tokens or 0) if usage else 0,
        ),
        model=model,
        finish_reason=FinishReason.STOP,
    )
