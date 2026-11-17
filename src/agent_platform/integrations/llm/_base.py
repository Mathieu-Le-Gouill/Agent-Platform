from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import AsyncIterator
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar, cast

from agent_platform.core.credentials import (
    ClientOptions,
    Credentials,
    resolve_max_retries,
    resolve_timeout,
)
from agent_platform.core.errors import PlatformError, ProviderError
from agent_platform.core.interfaces.llm.base import BaseLLMProvider, GenerationConfigT
from agent_platform.core.interfaces.llm.response import LLMResponse, StreamChunk
from agent_platform.core.observability import LLMCall, observe
from agent_platform.core.retry import retry_async, retry_sync
from agent_platform.core.schemas.message import Prompt
from agent_platform.core.schemas.token import TokenUsage

if TYPE_CHECKING:
    from agent_platform.agents.tools.base import Tool

_logger = logging.getLogger(__name__)

# Applied when neither `GenerationConfig.timeout` nor `ClientOptions.timeout`
# is set. A model that has not answered in two minutes is stuck, and the
# platform retry would otherwise wait on it forever.
DEFAULT_LLM_TIMEOUT_SECONDS = 120.0

_TRANSIENT_STATUS_CODES = frozenset({408, 409, 429})


def is_transient_status(status_code: int | None) -> bool:
    """Whether an HTTP status marks a temporary failure: a timeout, a conflict, a rate limit or a server error."""
    return status_code is not None and (
        status_code in _TRANSIENT_STATUS_CODES or status_code >= 500
    )


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
    each provider implements `stream()` directly but can still reuse `_observe`.
    """

    _provider_name: ClassVar[str]
    _missing_api_key_message: ClassVar[str] = "API key is required but was not provided"

    _credentials: Credentials
    _client_options: ClientOptions

    @abstractmethod
    def _default_config(self) -> GenerationConfigT: ...

    def _model_name(self, config: GenerationConfigT) -> str:
        return config.model

    def _observe(self, config: GenerationConfigT) -> AbstractContextManager[LLMCall]:
        return observe(
            LLMCall(
                provider=self._provider_name, request_model=self._model_name(config)
            )
        )

    @staticmethod
    def _record_usage(call: LLMCall, usage: TokenUsage) -> None:
        """Record a finished stream's accumulated usage on its observation.

        Streaming stays per-provider (see the class docstring), so this is the
        one piece of it the base can still own, keeping streamed and
        non-streamed calls comparable in a dashboard.
        """
        call.input_tokens = usage.input_tokens
        call.output_tokens = usage.output_tokens

    @classmethod
    def _record_response(cls, call: LLMCall, response: LLMResponse) -> None:
        """Record the response side of a chat call: what came back, not just what went out.

        Without this a trace shows a model was called but not which model
        answered, how the turn ended, or how many tool calls it asked for.
        """
        call.response_model = response.model
        call.response_id = response.response_id
        call.finish_reasons = (response.finish_reason.value,)
        if response.message is not None:
            call.tool_call_count = len(response.message.tool_calls)
        cls._record_usage(call, response.usage)

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

        def attempt() -> LLMResponse:
            return self._attempt_sync(prompt, config, tools)

        try:
            return retry_sync(attempt, max_attempts=self._max_attempts(config))
        except Exception:
            _logger.exception(
                "LLM generation failed in %s.generate", type(self).__qualname__
            )
            raise

    async def agenerate(
        self,
        prompt: Prompt,
        config: GenerationConfigT | None = None,
        tools: list[Tool] | None = None,
    ) -> LLMResponse:
        config = config or self._default_config()

        async def attempt() -> LLMResponse:
            return await self._attempt_async(prompt, config, tools)

        try:
            return await retry_async(attempt, max_attempts=self._max_attempts(config))
        except Exception:
            _logger.exception(
                "LLM generation failed in %s.agenerate", type(self).__qualname__
            )
            raise

    def _max_attempts(self, config: GenerationConfigT) -> int:
        return resolve_max_retries(config.max_retries, self._client_options) + 1

    def _attempt_sync(
        self, prompt: Prompt, config: GenerationConfigT, tools: list[Tool] | None
    ) -> LLMResponse:
        # The translation sits outside the observation, so the span records
        # the vendor exception and its type rather than the `ProviderError`.
        try:
            with self._observe(config) as call:
                client = self._sync_client(config)
                response = self._invoke_sync(client, prompt, config, tools)
                result = self._from_native(response, self._model_name(config))
                self._record_response(call, result)
                return result
        except PlatformError:
            raise
        except Exception as exc:
            raise self._translate(exc) from exc

    async def _attempt_async(
        self, prompt: Prompt, config: GenerationConfigT, tools: list[Tool] | None
    ) -> LLMResponse:
        try:
            with self._observe(config) as call:
                client = self._async_client(config)
                response = await self._invoke_async(client, prompt, config, tools)
                result = self._from_native(response, self._model_name(config))
                self._record_response(call, result)
                return result
        except PlatformError:
            raise
        except Exception as exc:
            raise self._translate(exc) from exc

    def _translate(self, exc: Exception) -> ProviderError:
        return ProviderError(
            f"LLM generation failed: {exc}", retryable=self._is_transient(exc)
        )

    def _is_transient(self, exc: Exception) -> bool:
        """Whether `exc` is a temporary failure worth another attempt.

        Subclasses extend this with their SDK's timeout, connection and
        HTTP-status errors. Anything not recognised is permanent, so a bad
        request or a revoked key fails at once instead of being retried.
        """
        return isinstance(exc, TimeoutError | ConnectionError)

    def _timeout(self, config: GenerationConfigT) -> float:
        """Return the request timeout in seconds: config, then client options, then the default."""
        timeout = resolve_timeout(config.timeout, self._client_options)
        return timeout if timeout is not None else DEFAULT_LLM_TIMEOUT_SECONDS

    @abstractmethod
    def stream(
        self,
        prompt: Prompt,
        config: GenerationConfigT | None = None,
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[StreamChunk]: ...
