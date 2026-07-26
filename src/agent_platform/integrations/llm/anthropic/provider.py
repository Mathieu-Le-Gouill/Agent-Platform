from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from anthropic import Anthropic, AsyncAnthropic

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
from agent_platform.integrations.credentials import AnthropicCredentials
from agent_platform.integrations.llm.anthropic.config import AnthropicGenerationConfig

if TYPE_CHECKING:
    from agent_platform.agents.tools.base import Tool


_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_THINKING_BUDGET = 5000
# Higher fallback so the default thinking_budget stays comfortably below
# max_tokens (Anthropic requires budget_tokens < max_tokens or the request 400s).
_DEFAULT_MAX_TOKENS_WITH_THINKING = 8192


class AnthropicLLM(BaseLLMProvider[AnthropicGenerationConfig]):
    def __init__(self, credentials: AnthropicCredentials | None = None) -> None:
        self._credentials = resolve_credentials(credentials, AnthropicCredentials)

    def _tool_to_schema(self, tool: Tool) -> dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": model_schema(tool.input_schema),
        }

    def _default_config(self) -> AnthropicGenerationConfig:
        return AnthropicGenerationConfig()

    def _client_kwargs(self, config: AnthropicGenerationConfig) -> dict[str, Any]:
        api_key = require_secret(
            self._credentials.api_key,
            "Anthropic API key is required but was not provided",
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

    def _client(self, config: AnthropicGenerationConfig) -> AsyncAnthropic:
        return AsyncAnthropic(**self._client_kwargs(config))

    def _sync_client(self, config: AnthropicGenerationConfig) -> Anthropic:
        return Anthropic(**self._client_kwargs(config))

    def generate(
        self,
        prompt: Prompt,
        config: AnthropicGenerationConfig | None = None,
        tools: list[Tool] | None = None,
    ) -> LLMResponse:
        config = config or self._default_config()

        with traced_operation_span(
            "chat",
            **{
                GenAIAttributes.PROVIDER_NAME: "anthropic",
                GenAIAttributes.REQUEST_MODEL: config.model,
            },
        ) as span:
            client = self._sync_client(config)
            system, messages = _to_native(prompt)
            params = _to_native_params(config)
            if system:
                params["system"] = system
            if tools:
                params["tools"] = [self._tool_to_schema(t) for t in tools]

            response = client.messages.create(
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
        config: AnthropicGenerationConfig | None = None,
        tools: list[Tool] | None = None,
    ) -> LLMResponse:
        config = config or self._default_config()

        with traced_operation_span(
            "chat",
            **{
                GenAIAttributes.PROVIDER_NAME: "anthropic",
                GenAIAttributes.REQUEST_MODEL: config.model,
            },
        ) as span:
            client = self._client(config)
            system, messages = _to_native(prompt)
            params = _to_native_params(config)
            if system:
                params["system"] = system
            if tools:
                params["tools"] = [self._tool_to_schema(t) for t in tools]

            response = await client.messages.create(
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
        config: AnthropicGenerationConfig | None = None,
    ) -> AsyncIterator[StreamChunk]:
        config = config or self._default_config()
        client = self._client(config)
        system, messages = _to_native(prompt)
        params = _to_native_params(config)
        if system:
            params["system"] = system

        with traced_operation_span(
            "chat",
            **{
                GenAIAttributes.PROVIDER_NAME: "anthropic",
                GenAIAttributes.REQUEST_MODEL: config.model,
            },
        ) as span:
            usage_totals = TokenUsage.zero()
            events: Any = await client.messages.create(
                model=config.model,
                messages=messages,  # type: ignore[arg-type]
                stream=True,
                **params,
            )
            async for raw_event in events:
                event: Any = raw_event
                if (
                    event.type == "content_block_delta"
                    and event.delta.type == "text_delta"
                ):
                    if event.delta.text:
                        yield StreamChunk(delta=event.delta.text)
                elif event.type == "message_start":
                    usage_totals = usage_totals + TokenUsage(
                        input_tokens=event.message.usage.input_tokens,
                        output_tokens=0,
                    )
                elif event.type == "message_delta" and event.usage is not None:
                    chunk_usage = TokenUsage(
                        input_tokens=0,
                        output_tokens=event.usage.output_tokens or 0,
                    )
                    usage_totals = usage_totals + chunk_usage
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
                return {"type": "image", "source": {"type": "url", "url": block.image}}
            mime = (
                f"image/{block.image.format.value}"
                if block.image.format
                else "image/png"
            )
            data = base64.b64encode(block.image.content).decode()
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": data},
            }
        case AudioBlock():
            raise ProviderError("Anthropic does not support audio content blocks")


def _content_to_native(m: ContentMessage) -> str | list[dict[str, Any]]:
    if isinstance(m.content, str):
        return m.content
    return [_block_to_native(b) for b in m.blocks]


def _to_native(prompt: Prompt) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []

    for m in prompt.messages:
        match m:
            case SystemMessage():
                if m.text:
                    system_parts.append(m.text)
            case UserMessage():
                messages.append({"role": "user", "content": _content_to_native(m)})
            case AssistantMessage():
                content = [_block_to_native(b) for b in m.blocks]
                for tc in m.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                messages.append({"role": "assistant", "content": content})
            case ToolMessage():
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.result.tool_call_id,
                                "content": m.result.content,
                                "is_error": m.result.is_error,
                            }
                        ],
                    }
                )

    return ("\n\n".join(system_parts) if system_parts else None, messages)


def _to_native_params(config: AnthropicGenerationConfig) -> dict[str, Any]:
    params: dict[str, Any] = {
        "temperature": config.temperature,
    }
    if config.top_p is not None:
        params["top_p"] = config.top_p
    if config.top_k is not None:
        params["top_k"] = config.top_k
    if config.stop_sequences:
        params["stop_sequences"] = config.stop_sequences

    output_config: dict[str, Any] = {}

    if config.effort is not None:
        # Modern effort control supersedes manual thinking_budget management;
        # skip the legacy `thinking` payload entirely when set.
        params["max_tokens"] = config.max_tokens or _DEFAULT_MAX_TOKENS
        output_config["effort"] = config.effort
    elif config.thinking:
        budget_tokens = config.thinking_budget or _DEFAULT_THINKING_BUDGET
        if config.max_tokens is not None:
            max_tokens = config.max_tokens
            if budget_tokens >= max_tokens:
                budget_tokens = max(1, max_tokens - 1024)
        else:
            # No explicit max_tokens: size it to comfortably fit the (possibly
            # user-supplied) budget rather than clamping the user's budget down
            # to an unrelated fixed default.
            max_tokens = max(_DEFAULT_MAX_TOKENS_WITH_THINKING, budget_tokens + 1024)
        params["max_tokens"] = max_tokens
        params["thinking"] = {
            "type": "enabled",
            "budget_tokens": budget_tokens,
        }
    else:
        params["max_tokens"] = config.max_tokens or _DEFAULT_MAX_TOKENS

    if config.cache_control:
        # Native top-level cache_control param on messages.create, no more
        # model_kwargs pass-through hack required by LangChain's ChatAnthropic.
        params["cache_control"] = {"type": "ephemeral"}

    match config.response_format:
        case ResponseFormat.TEXT:
            pass
        case ResponseFormat.JSON:
            # Anthropic's native structured-output mode (`output_config.format`)
            # only has a `json_schema` variant, no schema-less "json_object"
            # mode like OpenAI/Mistral; raise instead of silently ignoring the
            # request, since there's no way to honor it.
            raise ValueError(
                "Anthropic has no native schema-less JSON response format; "
                "use JSON_SCHEMA with an explicit json_schema instead"
            )
        case ResponseFormat.JSON_SCHEMA:
            if config.json_schema is None:
                raise ValueError(
                    "json_schema is required for JSON_SCHEMA response format"
                )
            output_config["format"] = {
                "type": "json_schema",
                "schema": config.json_schema,
            }

    if output_config:
        params["output_config"] = output_config

    params.update(config.extra_params)
    return params


def _from_native(response: Any, model: str) -> LLMResponse:
    tool_calls = [
        ToolCall(id=block.id, name=block.name, arguments=block.input)
        for block in response.content
        if block.type == "tool_use"
    ]

    content = "".join(block.text for block in response.content if block.type == "text")

    usage = response.usage

    return LLMResponse(
        message=AssistantMessage(content=content, tool_calls=tool_calls),
        usage=TokenUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        ),
        model=model,
        finish_reason=FinishReason.STOP,
    )
