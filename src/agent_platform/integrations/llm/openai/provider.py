from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, cast

from openai import AsyncOpenAI, AsyncStream, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

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
from agent_platform.integrations.credentials import OpenAICredentials
from agent_platform.integrations.llm._base import NativeLLMProvider
from agent_platform.integrations.llm.openai.config import OpenAIGenerationConfig
from agent_platform.integrations.llm.openai.mappers import (
    batch_job_from_native,
    from_native_response,
    to_native_messages,
    to_native_params,
)

if TYPE_CHECKING:
    from agent_platform.agents.tools.base import Tool


class OpenAILLM(
    NativeLLMProvider[OpenAIGenerationConfig, AsyncOpenAI, OpenAI, ChatCompletion],
    BaseBatchLLMProvider,
):
    _provider_name = "openai"
    _missing_api_key_message = "OPENAI API key is required but was not provided"

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

    def _async_client(self, config: OpenAIGenerationConfig) -> AsyncOpenAI:
        return AsyncOpenAI(**self._client_kwargs(config))

    def _sync_client(self, config: OpenAIGenerationConfig) -> OpenAI:
        return OpenAI(**self._client_kwargs(config))

    def _invoke_sync(
        self,
        client: OpenAI,
        prompt: Prompt,
        config: OpenAIGenerationConfig,
        tools: list[Tool] | None,
    ) -> ChatCompletion:
        messages = to_native_messages(prompt)
        params = to_native_params(config)
        if tools:
            params["tools"] = [self._tool_to_schema(t) for t in tools]

        return client.chat.completions.create(
            model=config.model,
            messages=messages,  # type: ignore[arg-type]
            **params,
        )

    async def _invoke_async(
        self,
        client: AsyncOpenAI,
        prompt: Prompt,
        config: OpenAIGenerationConfig,
        tools: list[Tool] | None,
    ) -> ChatCompletion:
        messages = to_native_messages(prompt)
        params = to_native_params(config)
        if tools:
            params["tools"] = [self._tool_to_schema(t) for t in tools]

        return await client.chat.completions.create(
            model=config.model,
            messages=messages,  # type: ignore[arg-type]
            **params,
        )

    def _from_native(self, response: ChatCompletion, model: str) -> LLMResponse:
        return from_native_response(response, model)

    async def stream(
        self,
        prompt: Prompt,
        config: OpenAIGenerationConfig | None = None,
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        config = config or self._default_config()
        client = self._async_client(config)
        messages = to_native_messages(prompt)
        params = to_native_params(config)
        params["stream_options"] = {"include_usage": True}
        if tools:
            params["tools"] = [self._tool_to_schema(t) for t in tools]

        with self._span(config) as span:
            usage_totals = TokenUsage.zero()
            events = cast(
                "AsyncStream[ChatCompletionChunk]",
                await client.chat.completions.create(
                    model=config.model,
                    messages=messages,  # type: ignore[arg-type]
                    stream=True,
                    **params,
                ),
            )
            async for chunk in events:
                if chunk.choices:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield StreamChunk(delta=delta)
                    tool_call_deltas = [
                        ToolCallDelta(
                            index=tc.index,
                            id=tc.id,
                            name=tc.function.name if tc.function else None,
                            arguments_delta=tc.function.arguments
                            if tc.function
                            else None,
                        )
                        for tc in (chunk.choices[0].delta.tool_calls or [])
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

    @error_logged(re_raise=ProviderError, message="Batch submission failed")
    @with_retry()
    async def submit_batch(self, requests: list[BatchRequest]) -> BatchJob:
        config = self._default_config()
        client = self._async_client(config)

        lines: list[str] = []
        for req in requests:
            req_config = cast(OpenAIGenerationConfig, req.config or config)
            body = {
                "model": req_config.model,
                "messages": to_native_messages(req.prompt),
                **to_native_params(req_config),
            }
            lines.append(
                json.dumps(
                    {
                        "custom_id": req.custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body,
                    }
                )
            )

        with traced_operation_span(
            "llm_batch_submit", **{GenAIAttributes.PROVIDER_NAME: self._provider_name}
        ):
            uploaded = await client.files.create(
                file=("batch.jsonl", "\n".join(lines).encode()),
                purpose="batch",
            )
            batch = await client.batches.create(
                input_file_id=uploaded.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
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
            batch = await client.batches.retrieve(batch_id)
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
            batch = await client.batches.retrieve(batch_id)
            if batch.output_file_id is None:
                return []
            content = await client.files.content(batch.output_file_id)
            results: list[BatchResult] = []
            for line in content.text.splitlines():
                if not line.strip():
                    continue
                raw = json.loads(line)
                error = raw.get("error")
                if error is not None:
                    results.append(
                        BatchResult(custom_id=raw["custom_id"], error=str(error))
                    )
                    continue
                completion = ChatCompletion.model_validate(raw["response"]["body"])
                results.append(
                    BatchResult(
                        custom_id=raw["custom_id"],
                        response=from_native_response(completion, completion.model),
                    )
                )
            return results
