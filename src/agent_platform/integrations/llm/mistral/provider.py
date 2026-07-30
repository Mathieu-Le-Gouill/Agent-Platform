from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from mistralai import Mistral
from mistralai.models import ChatCompletionResponse

from agent_platform.core.credentials import (
    ClientOptions,
    resolve_client_options,
    resolve_credentials,
    resolve_timeout,
)
from agent_platform.core.errors import require_secret
from agent_platform.core.genai_tracing import record_token_usage
from agent_platform.core.interfaces.llm.response import (
    FinishReason,
    LLMResponse,
    StreamChunk,
    ToolCallDelta,
)
from agent_platform.core.schemas import model_schema
from agent_platform.core.schemas.message import Prompt
from agent_platform.core.schemas.token import TokenUsage
from agent_platform.integrations.credentials import MistralCredentials
from agent_platform.integrations.llm._base import NativeLLMProvider
from agent_platform.integrations.llm.mistral.config import MistralGenerationConfig
from agent_platform.integrations.llm.mistral.mappers import (
    from_native_response,
    to_native_messages,
    to_native_params,
)

if TYPE_CHECKING:
    from agent_platform.agents.tools.base import Tool


class MistralLLM(
    NativeLLMProvider[MistralGenerationConfig, Mistral, Mistral, ChatCompletionResponse]
):
    _provider_name = "mistral"

    def __init__(
        self,
        credentials: MistralCredentials | None = None,
        client_options: ClientOptions | None = None,
    ) -> None:
        self._credentials = resolve_credentials(credentials, MistralCredentials)
        self._client_options = resolve_client_options(client_options)

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
        if self._client_options.base_url:
            kwargs["server_url"] = self._client_options.base_url

        timeout = resolve_timeout(config.timeout, self._client_options)
        if timeout is not None:
            kwargs["timeout_ms"] = int(timeout * 1000)

        # `config.max_retries`/`client_options.max_retries` are intentionally not wired
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
        messages = to_native_messages(prompt)
        params = to_native_params(config)
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
        messages = to_native_messages(prompt)
        params = to_native_params(config)
        if tools:
            params["tools"] = [self._tool_to_schema(t) for t in tools]

        return await client.chat.complete_async(
            model=config.model,
            messages=messages,  # type: ignore[arg-type]
            **params,
        )

    def _from_native(self, response: ChatCompletionResponse, model: str) -> LLMResponse:
        return from_native_response(response, model)

    async def stream(
        self,
        prompt: Prompt,
        config: MistralGenerationConfig | None = None,
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        config = config or self._default_config()
        client = self._async_client(config)
        messages = to_native_messages(prompt)
        params = to_native_params(config)
        if tools:
            params["tools"] = [self._tool_to_schema(t) for t in tools]

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
                    tool_call_deltas = [
                        ToolCallDelta(
                            index=idx,
                            id=tc.id,
                            name=tc.function.name if tc.function else None,
                            arguments_delta=(
                                tc.function.arguments
                                if tc.function
                                and isinstance(tc.function.arguments, str)
                                else None
                            ),
                        )
                        for idx, tc in enumerate(
                            chunk.choices[0].delta.tool_calls or []
                        )
                    ]
                    if tool_call_deltas:
                        yield StreamChunk(delta="", tool_call_deltas=tool_call_deltas)
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
