from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import httpx
from ollama import AsyncClient, ChatResponse, Client

from agent_platform.core.credentials import (
    ClientOptions,
    resolve_client_options,
    resolve_credentials,
    resolve_max_retries,
    resolve_timeout,
)
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
from agent_platform.integrations.credentials import OllamaCredentials
from agent_platform.integrations.llm._base import NativeLLMProvider
from agent_platform.integrations.llm.ollama.config import OllamaGenerationConfig
from agent_platform.integrations.llm.ollama.mappers import (
    from_native_response,
    to_native_messages,
    to_native_params,
)
from agent_platform.utils.env import from_env

if TYPE_CHECKING:
    from agent_platform.agents.tools.base import Tool

# Ollama has no auth concept, only a server location; when no client_options.base_url
# is given, fall back to the env var / the SDK's own local-daemon default.
_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaLLM(
    NativeLLMProvider[OllamaGenerationConfig, AsyncClient, Client, ChatResponse]
):
    _provider_name = "ollama"

    def __init__(
        self,
        credentials: OllamaCredentials | None = None,
        client_options: ClientOptions | None = None,
    ) -> None:
        self._credentials = resolve_credentials(credentials, OllamaCredentials)
        self._client_options = resolve_client_options(client_options)

    def _tool_to_schema(self, tool: Tool) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": model_schema(tool.input_schema),
                "required": [],
            },
        }

    def _default_config(self) -> OllamaGenerationConfig:
        return OllamaGenerationConfig()

    def _client_kwargs(self, config: OllamaGenerationConfig) -> dict[str, Any]:
        base_url = self._client_options.base_url or from_env(
            "OLLAMA_BASE_URL", _DEFAULT_BASE_URL
        )
        kwargs: dict[str, Any] = {"host": base_url}
        timeout = resolve_timeout(config.timeout, self._client_options)
        if timeout is not None:
            kwargs["timeout"] = timeout
        return kwargs

    def _max_retries(self, config: OllamaGenerationConfig) -> int:
        return resolve_max_retries(config.max_retries, self._client_options)

    def _async_client(self, config: OllamaGenerationConfig) -> AsyncClient:
        # `Client`/`AsyncClient` forward unrecognized kwargs straight to the
        # underlying `httpx` client, so `max_retries` is wired through a
        # custom transport rather than dropped. Unlike the platform's own
        # `@with_retry()` (full-request retry on any exception, used on
        # `agenerate`), `httpx`'s transport-level `retries` only covers
        # connection-establishment failures (`ConnectError`/`ConnectTimeout`),
        # not mid-response errors — but it's the only retry coverage `stream()`
        # and the sync `generate()` get, since neither carries `@with_retry()`.
        return AsyncClient(
            transport=httpx.AsyncHTTPTransport(retries=self._max_retries(config)),
            **self._client_kwargs(config),
        )

    def _sync_client(self, config: OllamaGenerationConfig) -> Client:
        return Client(
            transport=httpx.HTTPTransport(retries=self._max_retries(config)),
            **self._client_kwargs(config),
        )

    def _invoke_sync(
        self,
        client: Client,
        prompt: Prompt,
        config: OllamaGenerationConfig,
        tools: list[Tool] | None,
    ) -> ChatResponse:
        messages = to_native_messages(prompt)
        params = to_native_params(config)
        if tools:
            params["tools"] = [self._tool_to_schema(t) for t in tools]

        return client.chat(model=config.model, messages=messages, **params)

    async def _invoke_async(
        self,
        client: AsyncClient,
        prompt: Prompt,
        config: OllamaGenerationConfig,
        tools: list[Tool] | None,
    ) -> ChatResponse:
        messages = to_native_messages(prompt)
        params = to_native_params(config)
        if tools:
            params["tools"] = [self._tool_to_schema(t) for t in tools]

        return await client.chat(model=config.model, messages=messages, **params)

    def _from_native(self, response: ChatResponse, model: str) -> LLMResponse:
        return from_native_response(response, model)

    async def stream(
        self,
        prompt: Prompt,
        config: OllamaGenerationConfig | None = None,
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
            events = await client.chat(
                model=config.model, messages=messages, stream=True, **params
            )
            async for chunk in events:
                delta = chunk.message.content if chunk.message else None
                if delta:
                    yield StreamChunk(delta=delta)
                # Ollama does not fragment tool-call arguments across chunks
                # like OpenAI/Anthropic; each call arrives whole in one chunk.
                message_tool_calls = chunk.message.tool_calls if chunk.message else None
                if message_tool_calls:
                    yield StreamChunk(
                        delta="",
                        tool_call_deltas=[
                            ToolCallDelta(
                                index=idx,
                                id=uuid4().hex,
                                name=tc.function.name,
                                arguments_delta=json.dumps(dict(tc.function.arguments)),
                            )
                            for idx, tc in enumerate(message_tool_calls)
                        ],
                    )
                if chunk.done:
                    chunk_usage = TokenUsage(
                        input_tokens=chunk.prompt_eval_count or 0,
                        output_tokens=chunk.eval_count or 0,
                    )
                    usage_totals = chunk_usage
                    yield StreamChunk(
                        delta="",
                        finish_reason=FinishReason.STOP,
                        usage=chunk_usage,
                    )

            record_token_usage(span, usage_totals)
            yield StreamChunk(delta="", finish_reason=FinishReason.STOP)
