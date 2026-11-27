from __future__ import annotations

import functools
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import anthropic
from anthropic import Anthropic, AsyncAnthropic, AsyncStream
from anthropic.types import Message, RawMessageStreamEvent

from agent_platform.core.credentials import (
    ClientOptions,
    resolve_client_options,
    resolve_credentials,
    resolve_max_retries,
)
from agent_platform.core.errors import ProviderError, error_logged, require_secret
from agent_platform.core.interfaces.llm.batch import (
    BaseBatchLLMProvider,
    BatchJob,
    BatchRequest,
    BatchResult,
)
from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.interfaces.llm.response import (
    LLMResponse,
    StreamChunk,
    ToolCallDelta,
)
from agent_platform.core.observability import LLMBatch, observe
from agent_platform.core.retry import with_retry
from agent_platform.core.schemas import model_schema
from agent_platform.core.schemas.message import Prompt
from agent_platform.core.schemas.token import TokenUsage
from agent_platform.integrations.credentials import AnthropicCredentials
from agent_platform.integrations.llm._base import NativeLLMProvider
from agent_platform.integrations.llm._finish_reason import to_finish_reason
from agent_platform.integrations.llm.anthropic.config import AnthropicGenerationConfig
from agent_platform.integrations.llm.anthropic.mappers import (
    batch_job_from_native,
    from_native_response,
    to_native_messages,
    to_native_params,
)
from agent_platform.integrations.sdk_errors import (
    parse_retry_after,
    translate_sdk_error,
)

if TYPE_CHECKING:
    from agent_platform.core.interfaces.llm.tool_spec import ToolSpec

_BATCH_STATUS_ERROR = "Batch status check failed"
_BATCH_RESULTS_ERROR = "Batch result retrieval failed"
_translate_batch_status = functools.partial(
    translate_sdk_error, operation=_BATCH_STATUS_ERROR
)
_translate_batch_results = functools.partial(
    translate_sdk_error, operation=_BATCH_RESULTS_ERROR
)


class AnthropicLLM(
    NativeLLMProvider[AnthropicGenerationConfig, AsyncAnthropic, Anthropic, Message],
    BaseBatchLLMProvider,
):
    _provider_name = "anthropic"
    _missing_api_key_message = "Anthropic API key is required but was not provided"
    _api_key_env = "ANTHROPIC_API_KEY"
    _config_cls = AnthropicGenerationConfig

    def __init__(
        self,
        credentials: AnthropicCredentials | None = None,
        client_options: ClientOptions | None = None,
        *,
        model: str = "",
        options: Mapping[str, Any] | None = None,
    ) -> None:
        self._credentials = resolve_credentials(credentials, AnthropicCredentials)
        self._client_options = resolve_client_options(client_options)
        self._bind(model, options)

    def _tool_to_schema(self, tool: ToolSpec) -> dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": model_schema(tool.input_schema),
        }

    def _client_kwargs(
        self, config: AnthropicGenerationConfig, *, sdk_retries: int = 0
    ) -> dict[str, Any]:
        api_key = require_secret(
            self._credentials.api_key, self._missing_api_key_message
        )
        return {
            "api_key": api_key.get_secret_value(),
            "base_url": self._client_options.base_url,
            "max_retries": sdk_retries,
            "timeout": self._timeout(config),
        }

    # `generate`/`agenerate` retry on the platform side, so their clients make
    # no SDK retries; only the streaming client keeps them.
    def _async_client(self, config: AnthropicGenerationConfig) -> AsyncAnthropic:
        return AsyncAnthropic(**self._client_kwargs(config))

    def _sync_client(self, config: AnthropicGenerationConfig) -> Anthropic:
        return Anthropic(**self._client_kwargs(config))

    def _stream_client(self, config: AnthropicGenerationConfig) -> AsyncAnthropic:
        return AsyncAnthropic(
            **self._client_kwargs(
                config,
                sdk_retries=resolve_max_retries(
                    config.max_retries, self._client_options
                ),
            )
        )

    def _is_transient(self, exc: Exception) -> bool:
        # `APITimeoutError` is a subclass of `APIConnectionError`.
        if isinstance(exc, anthropic.APIConnectionError):
            return True
        return super()._is_transient(exc)

    def _status_code(self, exc: Exception) -> int | None:
        if isinstance(exc, anthropic.APIStatusError):
            return exc.status_code
        return None

    def _retry_after(self, exc: Exception) -> float | None:
        if isinstance(exc, anthropic.APIStatusError):
            return parse_retry_after(exc.response.headers.get("retry-after"))
        return None

    def _error_kinds(self, exc: Exception) -> frozenset[str]:
        # An error event inside a stream is an `APIStatusError` built with the
        # stream's own 200 response, so its status says nothing: the kind in
        # the event's body does.
        if isinstance(exc, anthropic.APIStatusError) and exc.type is not None:
            return frozenset({exc.type})
        return frozenset()

    def _is_context_window(self, exc: Exception) -> bool:
        return (
            isinstance(exc, anthropic.APIStatusError)
            and exc.status_code == 400
            and "prompt is too long" in exc.message.lower()
        )

    def _is_timeout(self, exc: Exception) -> bool:
        return isinstance(exc, anthropic.APITimeoutError) or super()._is_timeout(exc)

    def _invoke_sync(
        self,
        client: Anthropic,
        prompt: Prompt,
        config: AnthropicGenerationConfig,
        tools: Sequence[ToolSpec] | None,
    ) -> Message:
        system, messages = to_native_messages(prompt)
        params = to_native_params(config, has_tools=bool(tools))
        if system:
            params["system"] = system
        if tools:
            params["tools"] = [self._tool_to_schema(t) for t in tools]

        return client.messages.create(
            model=config.model,
            messages=messages,  # type: ignore[arg-type]
            **params,
        )

    async def _invoke_async(
        self,
        client: AsyncAnthropic,
        prompt: Prompt,
        config: AnthropicGenerationConfig,
        tools: Sequence[ToolSpec] | None,
    ) -> Message:
        system, messages = to_native_messages(prompt)
        params = to_native_params(config, has_tools=bool(tools))
        if system:
            params["system"] = system
        if tools:
            params["tools"] = [self._tool_to_schema(t) for t in tools]

        return await client.messages.create(
            model=config.model,
            messages=messages,  # type: ignore[arg-type]
            **params,
        )

    def _from_native(self, response: Message, model: str) -> LLMResponse:
        return from_native_response(response, model)

    async def _stream(
        self,
        prompt: Prompt,
        config: GenerationConfig | None = None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        resolved = self._resolve(config)
        client = self._stream_client(resolved)
        system, messages = to_native_messages(prompt)
        params = to_native_params(resolved, has_tools=bool(tools))
        if system:
            params["system"] = system
        if tools:
            params["tools"] = [self._tool_to_schema(t) for t in tools]

        with self._observe(resolved) as call:
            # `message_start` carries the input count; every `message_delta`
            # carries cumulative counts, so the last one wins.
            usage: TokenUsage | None = None
            stop_reason: str | None = None
            events = cast(
                "AsyncStream[RawMessageStreamEvent]",
                await client.messages.create(
                    model=resolved.model,
                    messages=messages,  # type: ignore[arg-type]
                    stream=True,
                    **params,
                ),
            )
            async for raw_event in events:
                # Each variant of the discriminated union exposes different
                # attributes; narrowing per-`type` would need an isinstance
                # chain over all 6, so defer to the runtime `.type` checks below.
                event: Any = raw_event
                if (
                    event.type == "content_block_delta"
                    and event.delta.type == "text_delta"
                ):
                    if event.delta.text:
                        yield StreamChunk(delta=event.delta.text)
                elif (
                    event.type == "content_block_start"
                    and event.content_block.type == "tool_use"
                ):
                    yield StreamChunk(
                        delta="",
                        tool_call_deltas=[
                            ToolCallDelta(
                                index=event.index,
                                id=event.content_block.id,
                                name=event.content_block.name,
                            )
                        ],
                    )
                elif (
                    event.type == "content_block_delta"
                    and event.delta.type == "input_json_delta"
                ):
                    yield StreamChunk(
                        delta="",
                        tool_call_deltas=[
                            ToolCallDelta(
                                index=event.index,
                                arguments_delta=event.delta.partial_json,
                            )
                        ],
                    )
                elif event.type == "message_start":
                    usage = TokenUsage(
                        input_tokens=event.message.usage.input_tokens or 0,
                        output_tokens=event.message.usage.output_tokens or 0,
                    )
                elif event.type == "message_delta":
                    if event.delta.stop_reason is not None:
                        stop_reason = event.delta.stop_reason
                    if event.usage is not None:
                        input_tokens = event.usage.input_tokens
                        if input_tokens is None:
                            input_tokens = usage.input_tokens if usage else 0
                        usage = TokenUsage(
                            input_tokens=input_tokens,
                            output_tokens=event.usage.output_tokens or 0,
                        )

            yield self._end_stream(call, usage, to_finish_reason(stop_reason))

    # Not retried: a failure after the request left may mean the batch was
    # created and will be billed, and a second submission would run it twice.
    @error_logged(re_raise=ProviderError, message="Batch submission failed")
    async def submit_batch(self, requests: list[BatchRequest]) -> BatchJob:
        client = self._async_client(self._resolve(None, require_model=False))

        batch_requests = []
        for req in requests:
            req_config = self._resolve(req.config)
            system, messages = to_native_messages(req.prompt)
            params = to_native_params(req_config)
            if system:
                params["system"] = system
            batch_requests.append(
                {
                    "custom_id": req.custom_id,
                    "params": {
                        "model": req_config.model,
                        "messages": messages,
                        **params,
                    },
                }
            )

        with observe(LLMBatch(stage="submit", provider=self._provider_name)):
            batch = await client.messages.batches.create(
                requests=batch_requests  # type: ignore[arg-type]
            )
            return batch_job_from_native(batch)

    @error_logged(re_raise=ProviderError, message=_BATCH_STATUS_ERROR)
    @with_retry(translate=_translate_batch_status)
    async def get_batch_status(self, batch_id: str) -> BatchJob:
        client = self._async_client(self._resolve(None, require_model=False))
        with observe(
            LLMBatch(stage="status", provider=self._provider_name, batch_id=batch_id)
        ):
            batch = await client.messages.batches.retrieve(batch_id)
            return batch_job_from_native(batch)

    @error_logged(re_raise=ProviderError, message=_BATCH_RESULTS_ERROR)
    @with_retry(translate=_translate_batch_results)
    async def fetch_batch_results(self, batch_id: str) -> list[BatchResult]:
        client = self._async_client(self._resolve(None, require_model=False))
        with observe(
            LLMBatch(stage="results", provider=self._provider_name, batch_id=batch_id)
        ):
            results: list[BatchResult] = []
            async for entry in await client.messages.batches.results(batch_id):
                if entry.result.type == "succeeded":
                    results.append(
                        BatchResult(
                            custom_id=entry.custom_id,
                            response=from_native_response(
                                entry.result.message, entry.result.message.model
                            ),
                        )
                    )
                else:
                    results.append(
                        BatchResult(
                            custom_id=entry.custom_id,
                            error=f"{entry.result.type}: {getattr(entry.result, 'error', '')}",
                        )
                    )
            return results
