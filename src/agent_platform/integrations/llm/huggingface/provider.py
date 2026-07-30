from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from huggingface_hub import AsyncInferenceClient, ChatCompletionOutput, InferenceClient

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
from agent_platform.integrations.credentials import HuggingFaceCredentials
from agent_platform.integrations.llm._base import NativeLLMProvider
from agent_platform.integrations.llm.huggingface.config import (
    HuggingFaceGenerationConfig,
)
from agent_platform.integrations.llm.huggingface.mappers import (
    from_native_response,
    to_native_messages,
    to_native_params,
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

    def __init__(
        self,
        credentials: HuggingFaceCredentials | None = None,
        client_options: ClientOptions | None = None,
    ) -> None:
        self._credentials = resolve_credentials(credentials, HuggingFaceCredentials)
        self._client_options = resolve_client_options(client_options)

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
        timeout = resolve_timeout(config.timeout, self._client_options)
        if timeout is not None:
            kwargs["timeout"] = timeout

        # `config.max_retries`/`client_options.max_retries` are intentionally not wired
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
        messages = to_native_messages(prompt)
        params = to_native_params(config)
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
        messages = to_native_messages(prompt)
        params = to_native_params(config)
        if tools:
            params["tools"] = [self._tool_to_schema(t) for t in tools]

        return await client.chat_completion(messages=messages, **params)

    def _from_native(self, response: ChatCompletionOutput, model: str) -> LLMResponse:
        return from_native_response(response, model)

    async def stream(
        self,
        prompt: Prompt,
        config: HuggingFaceGenerationConfig | None = None,
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
            events = await client.chat_completion(
                messages=messages, stream=True, **params
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
