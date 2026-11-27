from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AbstractContextManager
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast

from pydantic import ValidationError

from agent_platform.core.credentials import (
    ClientOptions,
    Credentials,
    resolve_max_retries,
    resolve_timeout,
)
from agent_platform.core.errors import ConfigError, PlatformError, ProviderError
from agent_platform.core.interfaces.llm.base import BaseLLMProvider, GenerationConfigT
from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.interfaces.llm.errors import (
    LLMContextWindowError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from agent_platform.core.interfaces.llm.response import LLMResponse, StreamChunk
from agent_platform.core.observability import LLMCall, observe
from agent_platform.core.retry import retry_async, retry_sync
from agent_platform.core.schemas.enums import FinishReason
from agent_platform.core.schemas.message import Prompt
from agent_platform.core.schemas.token import TokenUsage
from agent_platform.integrations.sdk_errors import is_transient_status

if TYPE_CHECKING:
    from agent_platform.core.interfaces.llm.tool_spec import ToolSpec

_logger = logging.getLogger(__name__)

# Applied when neither `GenerationConfig.timeout` nor `ClientOptions.timeout`
# is set. A model that has not answered in two minutes is stuck, and the
# platform retry would otherwise wait on it forever.
DEFAULT_LLM_TIMEOUT_SECONDS = 120.0

# Error kinds a vendor reports inside a stream that is already open (HTTP 200),
# where no status code is left to classify the failure by.
_TRANSIENT_ERROR_KINDS = frozenset({"overloaded_error", "api_error", "server_error"})
_RATE_LIMIT_ERROR_KINDS = frozenset({"rate_limit_error"})


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
    retry, error-translation and token-usage wiring that is otherwise
    identical across providers. The stream body is not templated: token-usage
    accumulation differs enough per vendor (e.g. Anthropic's input count in
    `message_start` and cumulative counts in `message_delta`, Gemini's
    cumulative usage repeated on every chunk, a single final-chunk usage
    elsewhere) that forcing one shape risks silently misreporting usage; each
    provider implements `_stream()` directly and ends it with `_end_stream`.
    Only error translation is templated: `stream()` wraps `_stream()` so a
    vendor failure, before or after the first chunk, reaches the caller as the
    same typed error `generate`/`agenerate` raise.

    A provider classifies its SDK's errors through the hooks `_status_code`,
    `_retry_after`, `_error_kinds`, `_is_context_window`, `_is_timeout` and
    `_is_transient`, and never overrides `stream()`.
    """

    _provider_name: ClassVar[str]
    _missing_api_key_message: ClassVar[str] = "API key is required but was not provided"
    # Name of the environment variable that holds the provider's API key, or
    # `None` for a provider that needs no key.
    _api_key_env: ClassVar[str | None] = None
    # The provider's own config class. Typed as the base class because a
    # `ClassVar` cannot hold a type variable; `_resolve` casts back.
    _config_cls: ClassVar[type[GenerationConfig]]

    _credentials: Credentials
    _client_options: ClientOptions
    # The model and the provider-only options this instance is bound to. They
    # are the lowest layers of every call's config (see `_resolve`).
    _model: str = ""
    _options: Mapping[str, Any] = MappingProxyType({})

    def missing_api_key_env(self) -> str | None:
        """Return the env var that should hold the API key when none is set, else `None`.

        A provider without a key (`_api_key_env` is `None`) never reports one
        missing. The key's value is never read out.
        """
        if self._api_key_env is None:
            return None
        if getattr(self._credentials, "api_key", None) is None:
            return self._api_key_env
        return None

    def _bind(self, model: str, options: Mapping[str, Any] | None) -> None:
        """Bind the instance to `model` and to provider-only `options`.

        Raises `ConfigError` naming the field, never its value, because
        options may hold sensitive values.
        """
        self._model = model
        self._options = MappingProxyType(self.validate_options(options or {}))

    @classmethod
    def validate_options(cls, options: Mapping[str, Any]) -> dict[str, Any]:
        """Return a copy of `options` once it is valid for this provider's config class.

        Raises `ConfigError` naming the bad field, never its value, because
        options may hold sensitive values.
        """
        fields = cls._config_cls.model_fields
        for key in options:
            if key == "model":
                raise ConfigError(
                    f"{cls._provider_name} LLM option 'model' is not allowed: "
                    "the model comes from the model string"
                )
            if key not in fields:
                raise ConfigError(f"Unknown {cls._provider_name} LLM option {key!r}")
        try:
            cls._config_cls.model_validate(dict(options))
        except ValidationError as exc:
            # Raised `from None`: the pydantic error text echoes the input value.
            bad_fields = sorted({".".join(map(str, e["loc"])) for e in exc.errors()})
            raise ConfigError(
                f"Invalid {cls._provider_name} LLM option(s): {', '.join(bad_fields)}"
            ) from None
        return dict(options)

    def _resolve(
        self, config: GenerationConfig | None, *, require_model: bool = True
    ) -> GenerationConfigT:
        """Return the provider's config for one call, from three layers.

        In order: the bound model, the bound options, then the fields the call
        config sets explicitly to a non-`None` value (an empty `model` counts
        as unset). A config of another provider class (a
        `FallbackLLMProvider` may forward one) only contributes the fields of
        the base `GenerationConfig`.
        """
        config_cls = self._config_cls
        merged: dict[str, Any] = {}
        if self._model:
            merged["model"] = self._model
        merged.update(self._options)
        if config is not None:
            readable = (
                config_cls.model_fields
                if isinstance(config, config_cls)
                else GenerationConfig.model_fields
            )
            for name in config.model_fields_set & readable.keys():
                value = getattr(config, name)
                if value is None or (name == "model" and not value):
                    continue
                merged[name] = value
        if require_model and not merged.get("model"):
            raise ConfigError(
                f"No model set for {self._provider_name}: use a "
                "'<provider>:<model>' string or set `model` in the config"
            )
        return cast(GenerationConfigT, config_cls(**merged))

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
    def _end_stream(
        cls,
        call: LLMCall,
        usage: TokenUsage | None,
        finish_reason: FinishReason,
    ) -> StreamChunk:
        """Record a finished stream on its observation and return its last chunk.

        The last chunk is the only one that carries usage: callers add up the
        usage of every chunk, so the call's total must reach them exactly once.
        """
        cls._record_usage(call, usage or TokenUsage.zero())
        call.finish_reasons = (finish_reason.value,)
        return StreamChunk(delta="", finish_reason=finish_reason, usage=usage)

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
        tools: Sequence[ToolSpec] | None,
    ) -> ResponseT:
        """Assemble the native request and make the blocking call, returning the raw response."""

    @abstractmethod
    async def _invoke_async(
        self,
        client: AsyncClientT,
        prompt: Prompt,
        config: GenerationConfigT,
        tools: Sequence[ToolSpec] | None,
    ) -> ResponseT:
        """Assemble the native request and make the async call, returning the raw response."""

    @abstractmethod
    def _from_native(self, response: ResponseT, model: str) -> LLMResponse: ...

    def generate(
        self,
        prompt: Prompt,
        config: GenerationConfig | None = None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> LLMResponse:
        resolved = self._resolve(config)

        def attempt() -> LLMResponse:
            return self._attempt_sync(prompt, resolved, tools)

        try:
            return retry_sync(attempt, max_attempts=self._max_attempts(resolved))
        except Exception as exc:
            self._log_failure(exc)
            raise

    async def agenerate(
        self,
        prompt: Prompt,
        config: GenerationConfig | None = None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> LLMResponse:
        resolved = self._resolve(config)

        async def attempt() -> LLMResponse:
            return await self._attempt_async(prompt, resolved, tools)

        try:
            return await retry_async(attempt, max_attempts=self._max_attempts(resolved))
        except Exception as exc:
            self._log_failure(exc)
            raise

    def _log_failure(self, exc: Exception) -> None:
        # One line, no traceback: the error travels up with its cause, and the
        # layer that finally handles it logs the traceback once.
        _logger.warning(
            "LLM call failed: provider=%s code=%s",
            self._provider_name,
            getattr(exc, "code", None),
        )

    def _max_attempts(self, config: GenerationConfigT) -> int:
        return resolve_max_retries(config.max_retries, self._client_options) + 1

    def _attempt_sync(
        self,
        prompt: Prompt,
        config: GenerationConfigT,
        tools: Sequence[ToolSpec] | None,
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
        self,
        prompt: Prompt,
        config: GenerationConfigT,
        tools: Sequence[ToolSpec] | None,
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
        """Return the platform error for a vendor failure, by the first rule that matches.

        A context-window overflow, a rate limit and a timeout each get their
        own type and code. Any other failure is `provider_unavailable` when it
        is temporary and `provider_bad_request` when it is not.
        """
        message = f"{self._provider_name} request failed: {exc}"
        if self._is_context_window(exc):
            return LLMContextWindowError(message, code="llm_context_window")
        rate_limited = self._status_code(exc) == 429 or bool(
            self._error_kinds(exc) & _RATE_LIMIT_ERROR_KINDS
        )
        if rate_limited:
            return LLMRateLimitError(
                message,
                code="llm_rate_limited",
                retryable=True,
                retry_after=self._retry_after(exc),
            )
        if self._is_timeout(exc):
            return LLMTimeoutError(message, code="llm_timeout", retryable=True)
        if self._is_transient(exc):
            return ProviderError(message, code="provider_unavailable", retryable=True)
        return ProviderError(message, code="provider_bad_request")

    def _is_transient(self, exc: Exception) -> bool:
        """Whether `exc` is a temporary failure worth another attempt.

        Subclasses extend this with their SDK's connection errors. Anything
        not recognised is permanent, so a bad request or a revoked key fails
        at once instead of being retried.
        """
        return (
            is_transient_status(self._status_code(exc))
            or bool(self._error_kinds(exc) & _TRANSIENT_ERROR_KINDS)
            or self._is_timeout(exc)
            or isinstance(exc, ConnectionError)
        )

    def _status_code(self, exc: Exception) -> int | None:
        """Return the HTTP status `exc` carries, or `None` when it has none."""
        return None

    def _retry_after(self, exc: Exception) -> float | None:
        """Return the seconds the vendor asked to wait before a retry, or `None`."""
        return None

    def _error_kinds(self, exc: Exception) -> frozenset[str]:
        """Return the error kinds in the body of `exc`, such as `overloaded_error`.

        A vendor error event inside an open stream has no failing status, so
        its body names the failure instead.
        """
        return frozenset()

    def _is_context_window(self, exc: Exception) -> bool:
        """Whether `exc` says the prompt does not fit the model's context window."""
        return False

    def _is_timeout(self, exc: Exception) -> bool:
        """Whether `exc` is a request that ran out of time."""
        return isinstance(exc, TimeoutError)

    def _timeout(self, config: GenerationConfigT) -> float:
        """Return the request timeout in seconds: config, then client options, then the default."""
        timeout = resolve_timeout(config.timeout, self._client_options)
        return timeout if timeout is not None else DEFAULT_LLM_TIMEOUT_SECONDS

    async def stream(
        self,
        prompt: Prompt,
        config: GenerationConfig | None = None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream the response; see `integrations/README.md` for the errors it raises."""
        # A client disconnect cancels the task: `asyncio.CancelledError` is a
        # `BaseException`, so it passes through untranslated.
        try:
            async for chunk in self._stream(prompt, config, tools):
                yield chunk
        except PlatformError:
            raise
        except Exception as exc:
            raise self._translate(exc) from exc

    @abstractmethod
    def _stream(
        self,
        prompt: Prompt,
        config: GenerationConfig | None = None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Make the streaming vendor call and yield its chunks, accumulating usage.

        Raise vendor errors as they are: `stream()` translates them.
        """
