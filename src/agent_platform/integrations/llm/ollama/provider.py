from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import httpx
from ollama import AsyncClient, ChatResponse, Client

from agent_platform.core.credentials import (
    resolve_credentials,
    resolve_max_retries,
    resolve_timeout,
)
from agent_platform.core.errors import ProviderError
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
from agent_platform.integrations.credentials import OllamaCredentials
from agent_platform.integrations.llm._base import NativeLLMProvider
from agent_platform.integrations.llm.ollama.config import OllamaGenerationConfig

if TYPE_CHECKING:
    from agent_platform.agents.tools.base import Tool


class OllamaLLM(
    NativeLLMProvider[OllamaGenerationConfig, AsyncClient, Client, ChatResponse]
):
    _provider_name = "ollama"

    def __init__(self, credentials: OllamaCredentials | None = None) -> None:
        self._credentials = resolve_credentials(credentials, OllamaCredentials)

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
        kwargs: dict[str, Any] = {"host": self._credentials.base_url}
        timeout = resolve_timeout(config.timeout, self._credentials)
        if timeout is not None:
            kwargs["timeout"] = timeout
        return kwargs

    def _max_retries(self, config: OllamaGenerationConfig) -> int:
        return resolve_max_retries(config.max_retries, self._credentials)

    def _client(self, config: OllamaGenerationConfig) -> AsyncClient:
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
        messages = _to_native(prompt)
        params = _to_native_params(config)
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
        messages = _to_native(prompt)
        params = _to_native_params(config)
        if tools:
            params["tools"] = [self._tool_to_schema(t) for t in tools]

        return await client.chat(model=config.model, messages=messages, **params)

    def _from_native(self, response: ChatResponse, model: str) -> LLMResponse:
        return _from_native(response, model)

    async def stream(
        self,
        prompt: Prompt,
        config: OllamaGenerationConfig | None = None,
    ) -> AsyncIterator[StreamChunk]:
        config = config or self._default_config()
        client = self._client(config)
        messages = _to_native(prompt)
        params = _to_native_params(config)

        with self._span(config) as span:
            usage_totals = TokenUsage.zero()
            events = await client.chat(
                model=config.model, messages=messages, stream=True, **params
            )
            async for chunk in events:
                delta = chunk.message.content if chunk.message else None
                if delta:
                    yield StreamChunk(delta=delta)
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


# --- Mappers ---


def _message_content(m: ContentMessage) -> tuple[str, list[str]]:
    if isinstance(m.content, str):
        return m.content, []

    text_parts: list[str] = []
    images: list[str] = []
    for block in m.blocks:
        match block:
            case TextBlock():
                text_parts.append(block.text)
            case ImageBlock():
                if isinstance(block.image, str):
                    images.append(block.image)
                else:
                    images.append(base64.b64encode(block.image.content).decode())
            case AudioBlock():
                raise ProviderError("Ollama does not support audio content blocks")

    return "".join(text_parts), images


def _to_native(prompt: Prompt) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for m in prompt.messages:
        match m:
            case SystemMessage():
                content, images = _message_content(m)
                entry: dict[str, Any] = {"role": "system", "content": content}
                if images:
                    entry["images"] = images
                result.append(entry)
            case UserMessage():
                content, images = _message_content(m)
                entry = {"role": "user", "content": content}
                if images:
                    entry["images"] = images
                result.append(entry)
            case AssistantMessage():
                content, images = _message_content(m)
                entry = {"role": "assistant", "content": content}
                if images:
                    entry["images"] = images
                if m.tool_calls:
                    entry["tool_calls"] = [
                        {"function": {"name": tc.name, "arguments": tc.arguments}}
                        for tc in m.tool_calls
                    ]
                result.append(entry)
            case ToolMessage():
                result.append(
                    {
                        "role": "tool",
                        "content": m.result.content,
                        "tool_name": m.result.name,
                    }
                )
    return result


def _to_native_params(config: OllamaGenerationConfig) -> dict[str, Any]:
    options: dict[str, Any] = {"temperature": config.temperature}
    if config.max_tokens is not None:
        options["num_predict"] = config.max_tokens
    if config.top_p is not None:
        options["top_p"] = config.top_p
    if config.top_k is not None:
        options["top_k"] = config.top_k
    if config.seed is not None:
        options["seed"] = config.seed
    if config.stop_sequences:
        options["stop"] = config.stop_sequences
    if config.repeat_penalty is not None:
        options["repeat_penalty"] = config.repeat_penalty
    if config.mirostat is not None:
        options["mirostat"] = config.mirostat
    if config.mirostat_tau is not None:
        options["mirostat_tau"] = config.mirostat_tau
    if config.mirostat_eta is not None:
        options["mirostat_eta"] = config.mirostat_eta
    if config.num_ctx is not None:
        options["num_ctx"] = config.num_ctx

    params: dict[str, Any] = {"options": options}

    match config.response_format:
        case ResponseFormat.JSON:
            params["format"] = "json"
        case ResponseFormat.JSON_SCHEMA:
            if config.json_schema is None:
                raise ValueError(
                    "json_schema is required for JSON_SCHEMA response format"
                )
            params["format"] = config.json_schema

    params.update(config.extra_params)
    return params


def _from_native(response: ChatResponse, model: str) -> LLMResponse:
    message = response.message

    tool_calls = [
        ToolCall(
            id=uuid4().hex,
            name=tc.function.name,
            arguments=dict(tc.function.arguments),
        )
        for tc in (message.tool_calls or [])
    ]

    return LLMResponse(
        message=AssistantMessage(content=message.content or "", tool_calls=tool_calls),
        usage=TokenUsage(
            input_tokens=response.prompt_eval_count or 0,
            output_tokens=response.eval_count or 0,
        ),
        model=model,
        finish_reason=FinishReason.STOP,
    )
