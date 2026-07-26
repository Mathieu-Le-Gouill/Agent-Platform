from __future__ import annotations

from abc import abstractmethod
from collections.abc import AsyncIterator
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast

from opentelemetry.trace import Span

from agent_platform.core.credentials import (
    ProviderCredentials,
    resolve_max_retries,
    resolve_timeout,
)
from agent_platform.core.errors import (
    ProviderError,
    error_logged,
    require_secret,
    with_retry,
)
from agent_platform.core.interfaces.llm.base import BaseLLMProvider, GenerationConfigT
from agent_platform.core.interfaces.llm.response import LLMResponse, StreamChunk
from agent_platform.core.schemas.message import Prompt
from agent_platform.core.tracing import (
    GenAIAttributes,
    record_token_usage,
    traced_operation_span,
)

if TYPE_CHECKING:
    from agent_platform.agents.tools.base import Tool

# The async client (e.g. `AsyncOpenAI`), used by `agenerate`/`_invoke_async`.
AsyncClientT = TypeVar("AsyncClientT")
# The sync client (e.g. `OpenAI`), used by `generate`/`_invoke_sync`. Equal to
# `AsyncClientT` for vendors with a single client class (e.g. Mistral's
# `Mistral`, whose methods split sync/async instead of the client itself).
SyncClientT = TypeVar("SyncClientT")
# The vendor's raw, non-streaming response object (e.g. `ChatCompletion`),
# shared by `_invoke_sync`/`_invoke_async` (same shape either way, only one
# is awaited) and consumed by `_from_native`.
ResponseT = TypeVar("ResponseT")


class NativeLLMProvider(
    BaseLLMProvider[GenerationConfigT],
    Generic[GenerationConfigT, AsyncClientT, SyncClientT, ResponseT],
):
    """Shared request/response plumbing for providers calling a vendor SDK directly.

    Every subclass owns its own vendor-specific pieces (client construction,
    message/param mapping, response parsing); this base owns the tracing,
    retry, and token-usage wiring that is otherwise identical across
    providers. `stream()` is intentionally not templated here: token-usage
    accumulation differs enough per vendor (e.g. Anthropic's additive
    `message_start`/`message_delta` events vs. a single final-chunk usage
    elsewhere) that forcing one shape risks silently misreporting usage;
    each provider implements `stream()` directly but can still reuse `_span`.
    """

    _provider_name: ClassVar[str]
    _missing_api_key_message: ClassVar[str] = "API key is required but was not provided"

    _credentials: ProviderCredentials

    @abstractmethod
    def _default_config(self) -> GenerationConfigT: ...

    def _model_name(self, config: GenerationConfigT) -> str:
        return config.model

    def _span(self, config: GenerationConfigT) -> AbstractContextManager[Span]:
        return traced_operation_span(
            "chat",
            **{
                GenAIAttributes.PROVIDER_NAME: self._provider_name,
                GenAIAttributes.REQUEST_MODEL: self._model_name(config),
            },
        )

    def _client_kwargs(self, config: GenerationConfigT) -> dict[str, Any]:
        api_key = require_secret(
            self._credentials.api_key, self._missing_api_key_message
        )
        kwargs: dict[str, Any] = {
            "api_key": api_key.get_secret_value(),
            "base_url": self._credentials.base_url,
            "max_retries": resolve_max_retries(config.max_retries, self._credentials),
        }
        timeout = resolve_timeout(config.timeout, self._credentials)
        if timeout is not None:
            kwargs["timeout"] = timeout
        return kwargs

    @abstractmethod
    def _async_client(self, config: GenerationConfigT) -> AsyncClientT: ...

    def _sync_client(self, config: GenerationConfigT) -> SyncClientT:
        # Default for vendors with one client class serving both sync and
        # async calls (e.g. Mistral); overridden where the SDK splits them
        # into distinct classes (e.g. `OpenAI`/`AsyncOpenAI`).
        return cast(SyncClientT, self._async_client(config))

    @abstractmethod
    def _invoke_sync(
        self,
        client: SyncClientT,
        prompt: Prompt,
        config: GenerationConfigT,
        tools: list[Tool] | None,
    ) -> ResponseT:
        """Assemble the native request and make the blocking call, returning the raw response."""

    @abstractmethod
    async def _invoke_async(
        self,
        client: AsyncClientT,
        prompt: Prompt,
        config: GenerationConfigT,
        tools: list[Tool] | None,
    ) -> ResponseT:
        """Assemble the native request and make the async call, returning the raw response."""

    @abstractmethod
    def _from_native(self, response: ResponseT, model: str) -> LLMResponse: ...

    def generate(
        self,
        prompt: Prompt,
        config: GenerationConfigT | None = None,
        tools: list[Tool] | None = None,
    ) -> LLMResponse:
        config = config or self._default_config()

        with self._span(config) as span:
            client = self._sync_client(config)
            response = self._invoke_sync(client, prompt, config, tools)
            result = self._from_native(response, self._model_name(config))
            record_token_usage(span, result.usage)
            return result

    @error_logged(re_raise=ProviderError, message="LLM generation failed")
    @with_retry()
    async def agenerate(
        self,
        prompt: Prompt,
        config: GenerationConfigT | None = None,
        tools: list[Tool] | None = None,
    ) -> LLMResponse:
        config = config or self._default_config()

        with self._span(config) as span:
            client = self._async_client(config)
            response = await self._invoke_async(client, prompt, config, tools)
            result = self._from_native(response, self._model_name(config))
            record_token_usage(span, result.usage)
            return result

    @abstractmethod
    def stream(
        self,
        prompt: Prompt,
        config: GenerationConfigT | None = None,
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[StreamChunk]: ...
