from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import httpx
from mistralai import Mistral
from mistralai.models import ChatCompletionResponse
from mistralai.models.mistralerror import MistralError

from agent_platform.core.credentials import (
    ClientOptions,
    resolve_client_options,
    resolve_credentials,
)
from agent_platform.core.errors import require_secret
from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.interfaces.llm.response import (
    LLMResponse,
    StreamChunk,
    ToolCallDelta,
)
from agent_platform.core.schemas import model_schema
from agent_platform.core.schemas.message import Prompt
from agent_platform.core.schemas.token import TokenUsage
from agent_platform.integrations.credentials import MistralCredentials
from agent_platform.integrations.llm._base import NativeLLMProvider
from agent_platform.integrations.llm._finish_reason import to_finish_reason
from agent_platform.integrations.llm.mistral.config import MistralGenerationConfig
from agent_platform.integrations.llm.mistral.mappers import (
    error_body,
    from_native_response,
    to_native_messages,
    to_native_params,
)
from agent_platform.integrations.sdk_errors import parse_retry_after

# Mistral's code for a prompt longer than the model's context window.
# https://docs.mistral.ai/resources/error-glossary (row 3059)
_PROMPT_TOO_LONG_CODE = "3059"
_PROMPT_TOO_LONG_TYPE = "invalid_request_prompt_too_long"

if TYPE_CHECKING:
    from agent_platform.core.interfaces.llm.tool_spec import ToolSpec


class MistralLLM(
    NativeLLMProvider[MistralGenerationConfig, Mistral, Mistral, ChatCompletionResponse]
):
    _provider_name = "mistral"
    _api_key_env = "MISTRAL_API_KEY"
    _config_cls = MistralGenerationConfig

    def __init__(
        self,
        credentials: MistralCredentials | None = None,
        client_options: ClientOptions | None = None,
        *,
        model: str = "",
        options: Mapping[str, Any] | None = None,
    ) -> None:
        self._credentials = resolve_credentials(credentials, MistralCredentials)
        self._client_options = resolve_client_options(client_options)
        self._bind(model, options)

    def _tool_to_schema(self, tool: ToolSpec) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": model_schema(tool.input_schema),
            },
        }

    def _async_client(self, config: MistralGenerationConfig) -> Mistral:
        api_key = require_secret(
            self._credentials.api_key,
            "Mistral API key is required but was not provided",
        )
        kwargs: dict[str, Any] = {
            "api_key": api_key.get_secret_value(),
            "timeout_ms": int(self._timeout(config) * 1000),
        }
        if self._client_options.base_url:
            kwargs["server_url"] = self._client_options.base_url

        # `config.max_retries`/`client_options.max_retries` are intentionally not wired
        # into the native SDK's `retry_config`: unlike OpenAI/Anthropic's simple
        # int attempt count, Mistral's RetryConfig is a time-based backoff
        # (initial_interval/max_interval/exponent/max_elapsed_time) with no
        # attempt-count knob, so there's no faithful translation. The platform
        # retry in `NativeLLMProvider.generate`/`agenerate` provides the
        # attempt-count-based retry instead.
        return Mistral(**kwargs)

    def _is_transient(self, exc: Exception) -> bool:
        if isinstance(exc, httpx.TransportError):
            return True
        return super()._is_transient(exc)

    def _status_code(self, exc: Exception) -> int | None:
        if isinstance(exc, MistralError):
            return exc.status_code
        return None

    def _retry_after(self, exc: Exception) -> float | None:
        if isinstance(exc, MistralError):
            return parse_retry_after(exc.headers.get("retry-after"))
        return None

    def _is_context_window(self, exc: Exception) -> bool:
        if not isinstance(exc, MistralError) or exc.status_code != 400:
            return False
        body = error_body(exc.body)
        return (
            str(body.get("code")) == _PROMPT_TOO_LONG_CODE
            or body.get("type") == _PROMPT_TOO_LONG_TYPE
        )

    def _is_timeout(self, exc: Exception) -> bool:
        return isinstance(exc, httpx.TimeoutException) or super()._is_timeout(exc)

    def _invoke_sync(
        self,
        client: Mistral,
        prompt: Prompt,
        config: MistralGenerationConfig,
        tools: Sequence[ToolSpec] | None,
    ) -> ChatCompletionResponse:
        messages = to_native_messages(prompt)
        params = to_native_params(config, has_tools=bool(tools))
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
        tools: Sequence[ToolSpec] | None,
    ) -> ChatCompletionResponse:
        messages = to_native_messages(prompt)
        params = to_native_params(config, has_tools=bool(tools))
        if tools:
            params["tools"] = [self._tool_to_schema(t) for t in tools]

        return await client.chat.complete_async(
            model=config.model,
            messages=messages,  # type: ignore[arg-type]
            **params,
        )

    def _from_native(self, response: ChatCompletionResponse, model: str) -> LLMResponse:
        return from_native_response(response, model)

    async def _stream(
        self,
        prompt: Prompt,
        config: GenerationConfig | None = None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        resolved = self._resolve(config)
        client = self._async_client(resolved)
        messages = to_native_messages(prompt)
        params = to_native_params(resolved, has_tools=bool(tools))
        if tools:
            params["tools"] = [self._tool_to_schema(t) for t in tools]

        with self._observe(resolved) as call:
            usage: TokenUsage | None = None
            raw_finish_reason: object = None
            # Stream-wide index per call id: the SDK's own `index` defaults
            # to 0 on every chunk, which would merge separate calls.
            call_indices: dict[str, int] = {}
            events = await client.chat.stream_async(
                model=resolved.model,
                messages=messages,  # type: ignore[arg-type]
                **params,
            )
            async for event in events:
                chunk = event.data
                if chunk.choices:
                    choice = chunk.choices[0]
                    if choice.finish_reason is not None:
                        raw_finish_reason = choice.finish_reason
                    delta = choice.delta.content
                    if isinstance(delta, str) and delta:
                        yield StreamChunk(delta=delta)
                    tool_call_deltas = []
                    for tc in choice.delta.tool_calls or []:
                        if tc.id and tc.id not in call_indices:
                            call_indices[tc.id] = len(call_indices)
                        # A fragment without an id continues the latest call.
                        index = (
                            call_indices[tc.id]
                            if tc.id
                            else max(len(call_indices) - 1, 0)
                        )
                        tool_call_deltas.append(
                            ToolCallDelta(
                                index=index,
                                id=tc.id,
                                name=tc.function.name if tc.function else None,
                                arguments_delta=(
                                    tc.function.arguments
                                    if tc.function
                                    and isinstance(tc.function.arguments, str)
                                    else None
                                ),
                            )
                        )
                    if tool_call_deltas:
                        yield StreamChunk(delta="", tool_call_deltas=tool_call_deltas)
                if chunk.usage is not None:
                    usage = TokenUsage(
                        input_tokens=chunk.usage.prompt_tokens or 0,
                        output_tokens=chunk.usage.completion_tokens or 0,
                    )

            yield self._end_stream(call, usage, to_finish_reason(raw_finish_reason))
