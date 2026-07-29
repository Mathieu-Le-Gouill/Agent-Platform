from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, cast

from anthropic import Anthropic, AsyncAnthropic, AsyncStream
from anthropic.types import Message, RawMessageStreamEvent

from agent_platform.core.credentials import resolve_credentials
from agent_platform.core.errors import ProviderError, error_logged, with_retry
from agent_platform.core.interfaces.llm.batch import (
    BaseBatchLLMProvider,
    BatchJob,
    BatchRequest,
    BatchResult,
)
from agent_platform.core.interfaces.llm.response import (
    FinishReason,
    LLMResponse,
    StreamChunk,
    ToolCallDelta,
)
from agent_platform.core.schemas import model_schema
from agent_platform.core.schemas.message import Prompt
from agent_platform.core.schemas.token import TokenUsage
from agent_platform.core.tracing import (
    GenAIAttributes,
    record_token_usage,
    traced_operation_span,
)
from agent_platform.integrations.credentials import AnthropicCredentials
from agent_platform.integrations.llm._base import NativeLLMProvider
from agent_platform.integrations.llm.anthropic.config import AnthropicGenerationConfig
from agent_platform.integrations.llm.anthropic.mappers import (
    batch_job_from_native,
    from_native_response,
    to_native_messages,
    to_native_params,
)

if TYPE_CHECKING:
    from agent_platform.agents.tools.base import Tool


class AnthropicLLM(
    NativeLLMProvider[AnthropicGenerationConfig, AsyncAnthropic, Anthropic, Message],
    BaseBatchLLMProvider,
):
    _provider_name = "anthropic"
    _missing_api_key_message = "Anthropic API key is required but was not provided"

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

    def _async_client(self, config: AnthropicGenerationConfig) -> AsyncAnthropic:
        return AsyncAnthropic(**self._client_kwargs(config))

    def _sync_client(self, config: AnthropicGenerationConfig) -> Anthropic:
        return Anthropic(**self._client_kwargs(config))

    def _invoke_sync(
        self,
        client: Anthropic,
        prompt: Prompt,
        config: AnthropicGenerationConfig,
        tools: list[Tool] | None,
    ) -> Message:
        system, messages = to_native_messages(prompt)
        params = to_native_params(config)
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
        tools: list[Tool] | None,
    ) -> Message:
        system, messages = to_native_messages(prompt)
        params = to_native_params(config)
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

    async def stream(
        self,
        prompt: Prompt,
        config: AnthropicGenerationConfig | None = None,
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        config = config or self._default_config()
        client = self._async_client(config)
        system, messages = to_native_messages(prompt)
        params = to_native_params(config)
        if system:
            params["system"] = system
        if tools:
            params["tools"] = [self._tool_to_schema(t) for t in tools]

        with self._span(config) as span:
            usage_totals = TokenUsage.zero()
            events = cast(
                "AsyncStream[RawMessageStreamEvent]",
                await client.messages.create(
                    model=config.model,
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

    @error_logged(re_raise=ProviderError, message="Batch submission failed")
    @with_retry()
    async def submit_batch(self, requests: list[BatchRequest]) -> BatchJob:
        config = self._default_config()
        client = self._async_client(config)

        batch_requests = []
        for req in requests:
            req_config = cast(AnthropicGenerationConfig, req.config or config)
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

        with traced_operation_span(
            "llm_batch_submit", **{GenAIAttributes.PROVIDER_NAME: self._provider_name}
        ):
            batch = await client.messages.batches.create(
                requests=batch_requests  # type: ignore[arg-type]
            )
            return batch_job_from_native(batch)

    @error_logged(re_raise=ProviderError, message="Batch status check failed")
    @with_retry()
    async def get_batch_status(self, batch_id: str) -> BatchJob:
        client = self._async_client(self._default_config())
        with traced_operation_span(
            "llm_batch_status",
            **{
                GenAIAttributes.PROVIDER_NAME: self._provider_name,
                GenAIAttributes.BATCH_ID: batch_id,
            },
        ):
            batch = await client.messages.batches.retrieve(batch_id)
            return batch_job_from_native(batch)

    @error_logged(re_raise=ProviderError, message="Batch result retrieval failed")
    @with_retry()
    async def fetch_batch_results(self, batch_id: str) -> list[BatchResult]:
        client = self._async_client(self._default_config())
        with traced_operation_span(
            "llm_batch_results",
            **{
                GenAIAttributes.PROVIDER_NAME: self._provider_name,
                GenAIAttributes.BATCH_ID: batch_id,
            },
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
