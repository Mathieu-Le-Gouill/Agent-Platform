from __future__ import annotations

import base64
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
    BatchStatus,
)
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
from agent_platform.core.tracing import (
    GenAIAttributes,
    record_token_usage,
    traced_operation_span,
)
from agent_platform.integrations.credentials import OpenAICredentials
from agent_platform.integrations.llm._base import NativeLLMProvider
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


_BATCH_STATUS_MAP: dict[str, BatchStatus] = {
    "validating": BatchStatus.PENDING,
    "in_progress": BatchStatus.IN_PROGRESS,
    "finalizing": BatchStatus.IN_PROGRESS,
    "cancelling": BatchStatus.IN_PROGRESS,
    "completed": BatchStatus.COMPLETED,
    "failed": BatchStatus.FAILED,
    "expired": BatchStatus.EXPIRED,
    "cancelled": BatchStatus.CANCELLED,
}


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
        messages = _to_native_messages(prompt)
        params = _to_native_params(config)
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
        messages = _to_native_messages(prompt)
        params = _to_native_params(config)
        if tools:
            params["tools"] = [self._tool_to_schema(t) for t in tools]

        return await client.chat.completions.create(
            model=config.model,
            messages=messages,  # type: ignore[arg-type]
            **params,
        )

    def _from_native(self, response: ChatCompletion, model: str) -> LLMResponse:
        return _from_native_response(response, model)

    async def stream(
        self,
        prompt: Prompt,
        config: OpenAIGenerationConfig | None = None,
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        config = config or self._default_config()
        client = self._async_client(config)
        messages = _to_native_messages(prompt)
        params = _to_native_params(config)
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
                "messages": _to_native_messages(req.prompt),
                **_to_native_params(req_config),
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
            return _batch_job_from_native(batch)

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
            return _batch_job_from_native(batch)

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
                        response=_from_native_response(completion, completion.model),
                    )
                )
            return results


def _batch_job_from_native(batch: Any) -> BatchJob:
    counts = getattr(batch, "request_counts", None)
    return BatchJob(
        id=batch.id,
        status=_BATCH_STATUS_MAP.get(batch.status, BatchStatus.PENDING),
        request_count=counts.total if counts is not None else None,
        completed_count=counts.completed if counts is not None else None,
    )


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


def _from_native_response(response: ChatCompletion, model: str) -> LLMResponse:
    message = response.choices[0].message

    tool_calls = [
        # `_tool_to_schema` only ever sends `"type": "function"` tools, so the
        # response only ever carries function-tool calls, never the SDK's
        # custom-tool-call variant.
        ToolCall(
            id=tc.id,
            name=tc.function.name,  # type: ignore[union-attr]
            arguments=json.loads(tc.function.arguments),  # type: ignore[union-attr]
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
