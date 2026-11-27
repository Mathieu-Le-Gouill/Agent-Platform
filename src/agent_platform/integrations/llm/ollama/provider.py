from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import httpx
from ollama import AsyncClient, ChatResponse, Client, ResponseError

from agent_platform.core.credentials import (
    ClientOptions,
    resolve_client_options,
    resolve_credentials,
    resolve_max_retries,
)
from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.interfaces.llm.response import (
    LLMResponse,
    StreamChunk,
    ToolCallDelta,
)
from agent_platform.core.schemas import model_schema
from agent_platform.core.schemas.message import Prompt
from agent_platform.core.schemas.token import TokenUsage
from agent_platform.integrations.credentials import OllamaCredentials
from agent_platform.integrations.llm._base import NativeLLMProvider
from agent_platform.integrations.llm._finish_reason import to_finish_reason
from agent_platform.integrations.llm.ollama.config import OllamaGenerationConfig
from agent_platform.integrations.llm.ollama.mappers import (
    from_native_response,
    to_native_messages,
    to_native_params,
)
from agent_platform.utils.env import from_env

if TYPE_CHECKING:
    from agent_platform.core.interfaces.llm.tool_spec import ToolSpec

# Ollama has no auth concept, only a server location; when no client_options.base_url
# is given, fall back to the env var / the SDK's own local-daemon default.
_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaLLM(
    NativeLLMProvider[OllamaGenerationConfig, AsyncClient, Client, ChatResponse]
):
    _provider_name = "ollama"
    _config_cls = OllamaGenerationConfig

    def __init__(
        self,
        credentials: OllamaCredentials | None = None,
        client_options: ClientOptions | None = None,
        *,
        model: str = "",
        options: Mapping[str, Any] | None = None,
    ) -> None:
        self._credentials = resolve_credentials(credentials, OllamaCredentials)
        self._client_options = resolve_client_options(client_options)
        self._bind(model, options)

    def _tool_to_schema(self, tool: ToolSpec) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": model_schema(tool.input_schema),
                "required": [],
            },
        }

    def _client_kwargs(self, config: OllamaGenerationConfig) -> dict[str, Any]:
        base_url = self._client_options.base_url or from_env(
            "OLLAMA_BASE_URL", _DEFAULT_BASE_URL
        )
        return {"host": base_url, "timeout": self._timeout(config)}

    def _max_retries(self, config: OllamaGenerationConfig) -> int:
        return resolve_max_retries(config.max_retries, self._client_options)

    # `Client`/`AsyncClient` forward unrecognized kwargs straight to the
    # underlying `httpx` client, so retries are set through a custom transport.
    # `generate`/`agenerate` retry on the platform side (a full-request retry
    # on any temporary failure), so their transports make no retries of their
    # own. `httpx`'s transport-level `retries` only covers connection
    # establishment (`ConnectError`/`ConnectTimeout`), not mid-response
    # errors, but it is the only retry coverage `stream()` gets, so the
    # streaming client keeps it.
    def _async_client(self, config: OllamaGenerationConfig) -> AsyncClient:
        return AsyncClient(
            transport=httpx.AsyncHTTPTransport(retries=0),
            **self._client_kwargs(config),
        )

    def _sync_client(self, config: OllamaGenerationConfig) -> Client:
        return Client(
            transport=httpx.HTTPTransport(retries=0),
            **self._client_kwargs(config),
        )

    def _stream_client(self, config: OllamaGenerationConfig) -> AsyncClient:
        return AsyncClient(
            transport=httpx.AsyncHTTPTransport(retries=self._max_retries(config)),
            **self._client_kwargs(config),
        )

    def _is_transient(self, exc: Exception) -> bool:
        if isinstance(exc, httpx.TransportError):
            return True
        return super()._is_transient(exc)

    def _status_code(self, exc: Exception) -> int | None:
        if isinstance(exc, ResponseError):
            return exc.status_code
        return None

    def _is_timeout(self, exc: Exception) -> bool:
        return isinstance(exc, httpx.TimeoutException) or super()._is_timeout(exc)

    def _invoke_sync(
        self,
        client: Client,
        prompt: Prompt,
        config: OllamaGenerationConfig,
        tools: Sequence[ToolSpec] | None,
    ) -> ChatResponse:
        messages = to_native_messages(prompt)
        params = to_native_params(config, has_tools=bool(tools))
        if tools:
            params["tools"] = [self._tool_to_schema(t) for t in tools]

        return client.chat(model=config.model, messages=messages, **params)

    async def _invoke_async(
        self,
        client: AsyncClient,
        prompt: Prompt,
        config: OllamaGenerationConfig,
        tools: Sequence[ToolSpec] | None,
    ) -> ChatResponse:
        messages = to_native_messages(prompt)
        params = to_native_params(config, has_tools=bool(tools))
        if tools:
            params["tools"] = [self._tool_to_schema(t) for t in tools]

        return await client.chat(model=config.model, messages=messages, **params)

    def _from_native(self, response: ChatResponse, model: str) -> LLMResponse:
        return from_native_response(response, model)

    async def _stream(
        self,
        prompt: Prompt,
        config: GenerationConfig | None = None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        resolved = self._resolve(config)
        client = self._stream_client(resolved)
        messages = to_native_messages(prompt)
        params = to_native_params(resolved, has_tools=bool(tools))
        if tools:
            params["tools"] = [self._tool_to_schema(t) for t in tools]

        with self._observe(resolved) as call:
            usage: TokenUsage | None = None
            raw_finish_reason: str | None = None
            tool_call_count = 0
            events = await client.chat(
                model=resolved.model, messages=messages, stream=True, **params
            )
            async for chunk in events:
                delta = chunk.message.content if chunk.message else None
                if delta:
                    yield StreamChunk(delta=delta)
                # Ollama does not fragment tool-call arguments across chunks
                # like OpenAI/Anthropic; each call arrives whole in one chunk,
                # so the index counts calls across the whole stream.
                message_tool_calls = chunk.message.tool_calls if chunk.message else None
                if message_tool_calls:
                    yield StreamChunk(
                        delta="",
                        tool_call_deltas=[
                            ToolCallDelta(
                                index=tool_call_count + offset,
                                id=uuid4().hex,
                                name=tc.function.name,
                                arguments_delta=json.dumps(dict(tc.function.arguments)),
                            )
                            for offset, tc in enumerate(message_tool_calls)
                        ],
                    )
                    tool_call_count += len(message_tool_calls)
                if chunk.done:
                    raw_finish_reason = chunk.done_reason
                    usage = TokenUsage(
                        input_tokens=chunk.prompt_eval_count or 0,
                        output_tokens=chunk.eval_count or 0,
                    )

            yield self._end_stream(call, usage, to_finish_reason(raw_finish_reason))
