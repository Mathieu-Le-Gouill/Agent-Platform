from __future__ import annotations

import functools
import json
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any, cast

import openai
from openai import AsyncOpenAI, AsyncStream, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

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
    FinishReason,
    LLMResponse,
    StreamChunk,
    ToolCallDelta,
)
from agent_platform.core.observability import LLMBatch, observe
from agent_platform.core.retry import with_retry
from agent_platform.core.schemas import model_schema
from agent_platform.core.schemas.message import Prompt
from agent_platform.core.schemas.token import TokenUsage
from agent_platform.integrations.credentials import OpenAICredentials
from agent_platform.integrations.llm._base import NativeLLMProvider
from agent_platform.integrations.llm.openai.config import OpenAIGenerationConfig
from agent_platform.integrations.llm.openai.mappers import (
    batch_job_from_native,
    from_native_response,
    strict_json_schema,
    to_native_messages,
    to_native_params,
)
from agent_platform.integrations.sdk_errors import (
    parse_retry_after,
    translate_sdk_error,
)

if TYPE_CHECKING:
    from agent_platform.agents.tools.base import Tool

_BATCH_STATUS_ERROR = "Batch status check failed"
_BATCH_RESULTS_ERROR = "Batch result retrieval failed"
_translate_batch_status = functools.partial(
    translate_sdk_error, operation=_BATCH_STATUS_ERROR
)
_translate_batch_results = functools.partial(
    translate_sdk_error, operation=_BATCH_RESULTS_ERROR
)


class OpenAILLM(
    NativeLLMProvider[OpenAIGenerationConfig, AsyncOpenAI, OpenAI, ChatCompletion],
    BaseBatchLLMProvider,
):
    _provider_name = "openai"
    _missing_api_key_message = "OPENAI API key is required but was not provided"
    _api_key_env = "OPENAI_API_KEY"
    _config_cls = OpenAIGenerationConfig

    def __init__(
        self,
        credentials: OpenAICredentials | None = None,
        client_options: ClientOptions | None = None,
        *,
        model: str = "",
        options: Mapping[str, Any] | None = None,
    ) -> None:
        self._credentials = resolve_credentials(credentials, OpenAICredentials)
        self._client_options = resolve_client_options(client_options)
        self._bind(model, options)

    def _tool_to_schema(self, tool: Tool) -> dict[str, Any]:
        parameters = model_schema(tool.input_schema)
        strict_parameters = strict_json_schema(parameters)
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": strict_parameters or parameters,
                "strict": strict_parameters is not None,
            },
        }

    def _client_kwargs(
        self, config: OpenAIGenerationConfig, *, sdk_retries: int = 0
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
    def _async_client(self, config: OpenAIGenerationConfig) -> AsyncOpenAI:
        return AsyncOpenAI(**self._client_kwargs(config))

    def _sync_client(self, config: OpenAIGenerationConfig) -> OpenAI:
        return OpenAI(**self._client_kwargs(config))

    def _stream_client(self, config: OpenAIGenerationConfig) -> AsyncOpenAI:
        return AsyncOpenAI(
            **self._client_kwargs(
                config,
                sdk_retries=resolve_max_retries(
                    config.max_retries, self._client_options
                ),
            )
        )

    def _is_transient(self, exc: Exception) -> bool:
        # `APITimeoutError` is a subclass of `APIConnectionError`.
        if isinstance(exc, openai.APIConnectionError):
            return True
        return super()._is_transient(exc)

    def _status_code(self, exc: Exception) -> int | None:
        if isinstance(exc, openai.APIStatusError):
            return exc.status_code
        return None

    def _retry_after(self, exc: Exception) -> float | None:
        if isinstance(exc, openai.APIStatusError):
            return parse_retry_after(exc.response.headers.get("retry-after"))
        return None

    def _error_kinds(self, exc: Exception) -> frozenset[str]:
        # An error event inside a stream is a bare `APIError` built from the
        # event's body, so its kind is all there is to classify it by.
        if isinstance(exc, openai.APIError):
            return frozenset(k for k in (exc.type, exc.code) if isinstance(k, str))
        return frozenset()

    def _is_context_window(self, exc: Exception) -> bool:
        return (
            isinstance(exc, openai.APIError) and exc.code == "context_length_exceeded"
        )

    def _is_timeout(self, exc: Exception) -> bool:
        return isinstance(exc, openai.APITimeoutError) or super()._is_timeout(exc)

    def _invoke_sync(
        self,
        client: OpenAI,
        prompt: Prompt,
        config: OpenAIGenerationConfig,
        tools: list[Tool] | None,
    ) -> ChatCompletion:
        messages = to_native_messages(prompt)
        params = to_native_params(config, has_tools=bool(tools))
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
        params = to_native_params(config, has_tools=bool(tools))
        if tools:
            params["tools"] = [self._tool_to_schema(t) for t in tools]

        return await client.chat.completions.create(
            model=config.model,
            messages=messages,  # type: ignore[arg-type]
            **params,
        )

    def _from_native(self, response: ChatCompletion, model: str) -> LLMResponse:
        return from_native_response(response, model)

    async def _stream(
        self,
        prompt: Prompt,
        config: GenerationConfig | None = None,
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        resolved = self._resolve(config)
        client = self._stream_client(resolved)
        messages = to_native_messages(prompt)
        params = to_native_params(resolved, has_tools=bool(tools))
        params["stream_options"] = {"include_usage": True}
        if tools:
            params["tools"] = [self._tool_to_schema(t) for t in tools]

        with self._observe(resolved) as call:
            usage_totals = TokenUsage.zero()
            events = cast(
                "AsyncStream[ChatCompletionChunk]",
                await client.chat.completions.create(
                    model=resolved.model,
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

            self._record_usage(call, usage_totals)
            yield StreamChunk(delta="", finish_reason=FinishReason.STOP)

    # Not retried: a failure after the request left may mean the batch was
    # created and will be billed, and a second submission would run it twice.
    @error_logged(re_raise=ProviderError, message="Batch submission failed")
    async def submit_batch(self, requests: list[BatchRequest]) -> BatchJob:
        client = self._async_client(self._resolve(None, require_model=False))

        lines: list[str] = []
        for req in requests:
            req_config = self._resolve(req.config)
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

        with observe(LLMBatch(stage="submit", provider=self._provider_name)):
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

    @error_logged(re_raise=ProviderError, message=_BATCH_STATUS_ERROR)
    @with_retry(translate=_translate_batch_status)
    async def get_batch_status(self, batch_id: str) -> BatchJob:
        client = self._async_client(self._resolve(None, require_model=False))
        with observe(
            LLMBatch(stage="status", provider=self._provider_name, batch_id=batch_id)
        ):
            batch = await client.batches.retrieve(batch_id)
            return batch_job_from_native(batch)

    @error_logged(re_raise=ProviderError, message=_BATCH_RESULTS_ERROR)
    @with_retry(translate=_translate_batch_results)
    async def fetch_batch_results(self, batch_id: str) -> list[BatchResult]:
        client = self._async_client(self._resolve(None, require_model=False))
        with observe(
            LLMBatch(stage="results", provider=self._provider_name, batch_id=batch_id)
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
