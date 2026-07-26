from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI, OpenAI

from agent_platform.core.credentials import (
    resolve_credentials,
    resolve_max_retries,
    resolve_timeout,
)
from agent_platform.core.errors import (
    ProviderError,
    error_logged,
    require_secret,
    with_retry,
)
from agent_platform.core.interfaces.llm.base import BaseLLMProvider
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
from agent_platform.core.tracing import (
    GenAIAttributes,
    record_token_usage,
    traced_operation_span,
)
from agent_platform.integrations.credentials import OpenAICredentials
from agent_platform.integrations.llm.openai.config import OpenAIGenerationConfig

if TYPE_CHECKING:
    from agent_platform.agents.tools.base import Tool


# Reasoning models (o-series, gpt-5 non-chat) reject non-default temperature/top_p
# and only accept reasoning_effort on this family. https://platform.openai.com/docs/guides/reasoning
_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4-mini")
_DEFAULT_TEMPERATURE: float = OpenAIGenerationConfig.model_fields["temperature"].default


def _is_reasoning_model(model: str) -> bool:
    model_lower = model.lower()
    if model_lower.startswith(_REASONING_MODEL_PREFIXES):
        return True
    return model_lower.startswith("gpt-5") and "chat" not in model_lower


class OpenAILLM(BaseLLMProvider[OpenAIGenerationConfig]):
    def __init__(self, credentials: OpenAICredentials | None = None) -> None:
        self._credentials = resolve_credentials(credentials, OpenAICredentials)

    def _tool_to_schema(self, tool: Tool) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": model_schema(tool.input_schema),
                "strict": True,
            },
        }

    def _default_config(self) -> OpenAIGenerationConfig:
        return OpenAIGenerationConfig()

    def _client_kwargs(self, config: OpenAIGenerationConfig) -> dict[str, Any]:
        api_key = require_secret(
            self._credentials.api_key,
            "OPENAI API key is required but was not provided",
        )
        kwargs: dict[str, Any] = {
            "api_key": api_key.get_secret_value(),
            "base_url": self._credentials.base_url,
            "max_retries": resolve_max_retries(config.max_retries, self._credentials),
        }
        timeout = resolve_timeout(config.timeout, self._credentials)
        if timeout is not None:
            kwargs["timeout"] = timeout
        return kwargs

    def _client(self, config: OpenAIGenerationConfig) -> AsyncOpenAI:
        return AsyncOpenAI(**self._client_kwargs(config))

    def _sync_client(self, config: OpenAIGenerationConfig) -> OpenAI:
        return OpenAI(**self._client_kwargs(config))

    def generate(
        self,
        prompt: Prompt,
        config: OpenAIGenerationConfig | None = None,
        tools: list[Tool] | None = None,
    ) -> LLMResponse:
        config = config or self._default_config()

        with traced_operation_span(
            "chat",
            **{
                GenAIAttributes.PROVIDER_NAME: "openai",
                GenAIAttributes.REQUEST_MODEL: config.model,
            },
        ) as span:
            client = self._sync_client(config)
            messages = _to_native(prompt)
            params = _to_native_params(config)
            if tools:
                params["tools"] = [self._tool_to_schema(t) for t in tools]

            response = client.chat.completions.create(
                model=config.model,
                messages=messages,  # type: ignore[arg-type]
                **params,
            )
            result = _from_native(response, config.model)
            record_token_usage(span, result.usage)
            return result

    @error_logged(re_raise=ProviderError, message="LLM generation failed")
    @with_retry()
    async def agenerate(
        self,
        prompt: Prompt,
        config: OpenAIGenerationConfig | None = None,
        tools: list[Tool] | None = None,
    ) -> LLMResponse:
        config = config or self._default_config()

        with traced_operation_span(
            "chat",
            **{
                GenAIAttributes.PROVIDER_NAME: "openai",
                GenAIAttributes.REQUEST_MODEL: config.model,
            },
        ) as span:
            client = self._client(config)
            messages = _to_native(prompt)
            params = _to_native_params(config)
            if tools:
                params["tools"] = [self._tool_to_schema(t) for t in tools]

            response = await client.chat.completions.create(
                model=config.model,
                messages=messages,  # type: ignore[arg-type]
                **params,
            )
            result = _from_native(response, config.model)
            record_token_usage(span, result.usage)
            return result

    async def stream(
        self,
        prompt: Prompt,
        config: OpenAIGenerationConfig | None = None,
    ) -> AsyncIterator[StreamChunk]:
        config = config or self._default_config()
        client = self._client(config)
        messages = _to_native(prompt)
        params = _to_native_params(config)
        params["stream_options"] = {"include_usage": True}

        with traced_operation_span(
            "chat",
            **{
                GenAIAttributes.PROVIDER_NAME: "openai",
                GenAIAttributes.REQUEST_MODEL: config.model,
            },
        ) as span:
            usage_totals = TokenUsage.zero()
            events: Any = await client.chat.completions.create(
                model=config.model,
                messages=messages,  # type: ignore[arg-type]
                stream=True,
                **params,
            )
            async for chunk in events:
                if chunk.choices:
                    delta = chunk.choices[0].delta.content
                    if delta:
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
            return {"type": "image_url", "image_url": {"url": url}}
        case AudioBlock():
            data = base64.b64encode(block.audio.content).decode()
            fmt = block.audio.format.value if block.audio.format else "wav"
            return {"type": "input_audio", "input_audio": {"data": data, "format": fmt}}


def _content_to_native(m: ContentMessage) -> str | list[dict[str, Any]]:
    if isinstance(m.content, str):
        return m.content
    return [_block_to_native(b) for b in m.blocks]


def _to_native(prompt: Prompt) -> list[dict[str, Any]]:
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
                    }
                )
    return result


def _to_native_params(config: OpenAIGenerationConfig) -> dict[str, Any]:
    reasoning_model = _is_reasoning_model(config.model)

    params: dict[str, Any] = {}
    if not reasoning_model and config.temperature != _DEFAULT_TEMPERATURE:
        params["temperature"] = config.temperature
    if config.max_tokens is not None:
        params["max_tokens"] = config.max_tokens
    if config.top_p is not None:
        params["top_p"] = config.top_p
    if config.stop_sequences:
        params["stop"] = config.stop_sequences
    if config.seed is not None:
        params["seed"] = config.seed

    if config.frequency_penalty is not None:
        params["frequency_penalty"] = config.frequency_penalty
    if config.presence_penalty is not None:
        params["presence_penalty"] = config.presence_penalty

    if config.reasoning_effort is not None and reasoning_model:
        params["reasoning_effort"] = config.reasoning_effort
    if not config.parallel_tool_calls:
        params["parallel_tool_calls"] = config.parallel_tool_calls

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
                    "name": "response",
                    "schema": config.json_schema,
                    "strict": config.strict,
                },
            }

    params.update(config.extra_params)
    return params


def _from_native(response: Any, model: str) -> LLMResponse:
    message = response.choices[0].message

    tool_calls = [
        ToolCall(
            id=tc.id,
            name=tc.function.name,
            arguments=json.loads(tc.function.arguments),
        )
        for tc in (message.tool_calls or [])
    ]

    usage = response.usage

    return LLMResponse(
        message=AssistantMessage(content=message.content or "", tool_calls=tool_calls),
        usage=TokenUsage(
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        ),
        model=model,
        finish_reason=FinishReason.STOP,
    )
