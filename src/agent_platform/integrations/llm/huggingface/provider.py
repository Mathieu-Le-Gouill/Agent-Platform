from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from huggingface_hub import AsyncInferenceClient, ChatCompletionOutput, InferenceClient

from agent_platform.core.credentials import resolve_credentials, resolve_timeout
from agent_platform.core.errors import ProviderError, require_secret
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
from agent_platform.integrations.credentials import HuggingFaceCredentials
from agent_platform.integrations.llm._base import NativeLLMProvider
from agent_platform.integrations.llm.huggingface.config import (
    HuggingFaceGenerationConfig,
)

if TYPE_CHECKING:
    from agent_platform.agents.tools.base import Tool


class HuggingFaceLLM(
    NativeLLMProvider[
        HuggingFaceGenerationConfig,
        AsyncInferenceClient,
        InferenceClient,
        ChatCompletionOutput,
    ]
):
    _provider_name = "huggingface"

    def __init__(self, credentials: HuggingFaceCredentials | None = None) -> None:
        self._credentials = resolve_credentials(credentials, HuggingFaceCredentials)

    def _model_name(self, config: HuggingFaceGenerationConfig) -> str:
        return config.repo_id

    def _tool_to_schema(self, tool: Tool) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": model_schema(tool.input_schema),
            },
        }

    def _default_config(self) -> HuggingFaceGenerationConfig:
        return HuggingFaceGenerationConfig()

    def _client_kwargs(self, config: HuggingFaceGenerationConfig) -> dict[str, Any]:
        api_key = require_secret(
            self._credentials.api_key,
            "Hugging Face Hub API token is required but was not provided",
        )
        kwargs: dict[str, Any] = {
            "model": config.repo_id,
            "provider": config.provider,
            "token": api_key.get_secret_value(),
        }
        timeout = resolve_timeout(config.timeout, self._credentials)
        if timeout is not None:
            kwargs["timeout"] = timeout

        # `config.max_retries`/`credentials.max_retries` are intentionally not wired
        # in: `InferenceClient`/`AsyncInferenceClient` have no built-in retry-count
        # knob. The platform's own `@with_retry()` decorator on `agenerate` already
        # provides equivalent attempt-count-based retry behavior.
        return kwargs

    def _async_client(
        self, config: HuggingFaceGenerationConfig
    ) -> AsyncInferenceClient:
        return AsyncInferenceClient(**self._client_kwargs(config))

    def _sync_client(self, config: HuggingFaceGenerationConfig) -> InferenceClient:
        return InferenceClient(**self._client_kwargs(config))

    def _invoke_sync(
        self,
        client: InferenceClient,
        prompt: Prompt,
        config: HuggingFaceGenerationConfig,
        tools: list[Tool] | None,
    ) -> ChatCompletionOutput:
        messages = _to_native_messages(prompt)
        params = _to_native_params(config)
        if tools:
            params["tools"] = [self._tool_to_schema(t) for t in tools]

        return client.chat_completion(messages=messages, **params)

    async def _invoke_async(
        self,
        client: AsyncInferenceClient,
        prompt: Prompt,
        config: HuggingFaceGenerationConfig,
        tools: list[Tool] | None,
    ) -> ChatCompletionOutput:
        messages = _to_native_messages(prompt)
        params = _to_native_params(config)
        if tools:
            params["tools"] = [self._tool_to_schema(t) for t in tools]

        return await client.chat_completion(messages=messages, **params)

    def _from_native(self, response: ChatCompletionOutput, model: str) -> LLMResponse:
        return _from_native_response(response, model)

    async def stream(
        self,
        prompt: Prompt,
        config: HuggingFaceGenerationConfig | None = None,
    ) -> AsyncIterator[StreamChunk]:
        config = config or self._default_config()
        client = self._async_client(config)
        messages = _to_native_messages(prompt)
        params = _to_native_params(config)

        with self._span(config) as span:
            usage_totals = TokenUsage.zero()
            events = await client.chat_completion(
                messages=messages, stream=True, **params
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
            raise ProviderError("Hugging Face does not support audio content blocks")


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


def _to_native_params(config: HuggingFaceGenerationConfig) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if config.temperature is not None:
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
                },
            }

    # These text-generation-task knobs (from the retired `HuggingFaceEndpoint`
    # raw-completion API) have no equivalent top-level field in the OpenAI-
    # compatible chat_completion schema; forward them best-effort via
    # `extra_body`, which providers behind the Hub's inference API may honor.
    extra_body: dict[str, Any] = {}
    if config.top_k is not None:
        extra_body["top_k"] = config.top_k
    if config.repetition_penalty is not None:
        extra_body["repetition_penalty"] = config.repetition_penalty
    if config.do_sample is not None:
        extra_body["do_sample"] = config.do_sample
    if config.typical_p is not None:
        extra_body["typical_p"] = config.typical_p
    if config.return_full_text is not None:
        extra_body["return_full_text"] = config.return_full_text
    if extra_body:
        params["extra_body"] = extra_body

    params.update(config.extra_params)
    return params


def _from_native_response(response: ChatCompletionOutput, model: str) -> LLMResponse:
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
