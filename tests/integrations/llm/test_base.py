import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from unittest.mock import AsyncMock, MagicMock

import anthropic
import httpx
import openai
import pytest
from google.genai import errors as genai_errors
from huggingface_hub.errors import HfHubHTTPError
from mistralai.models.mistralerror import MistralError
from ollama import ResponseError
from pydantic import SecretStr

from agent_platform.core.credentials import ClientOptions, Credentials
from agent_platform.core.errors import ConfigError, ProviderError
from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.interfaces.llm.errors import (
    LLMContextWindowError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from agent_platform.core.interfaces.llm.response import LLMResponse, StreamChunk
from agent_platform.core.schemas.message import Prompt, UserMessage
from agent_platform.integrations.credentials import (
    AnthropicCredentials,
    GoogleCredentials,
    HuggingFaceCredentials,
    MistralCredentials,
    OllamaCredentials,
    OpenAICredentials,
)
from agent_platform.integrations.llm._base import (
    DEFAULT_LLM_TIMEOUT_SECONDS,
    NativeLLMProvider,
)
from agent_platform.integrations.llm.anthropic.provider import AnthropicLLM
from agent_platform.integrations.llm.openai.provider import OpenAILLM
from agent_platform.integrations.sdk_errors import (
    is_transient_status,
    parse_retry_after,
)
from tests.helpers import make_fake_llm_response

_BASE_LOGGER = "agent_platform.integrations.llm._base"


class VendorBadRequest(Exception):
    """Stands for a permanent vendor error such as an HTTP 400."""


def _final_failure_records(caplog) -> list[logging.LogRecord]:
    """The records of the one line a failed call logs (retries log theirs apart)."""
    return [
        r
        for r in caplog.records
        if r.name == _BASE_LOGGER and r.getMessage().startswith("LLM call failed")
    ]


class FakeProvider(
    NativeLLMProvider[GenerationConfig, object, object, LLMResponse],
):
    """A provider whose vendor calls raise the scripted errors, then succeed."""

    _provider_name = "fake"
    _config_cls = GenerationConfig

    def __init__(
        self,
        errors: list[Exception],
        client_options: ClientOptions | None = None,
        *,
        model: str = "fake-model",
        options: dict | None = None,
    ) -> None:
        self._credentials = Credentials()
        self._client_options = client_options or ClientOptions()
        self._errors = list(errors)
        self.vendor_calls = 0
        self.seen_configs: list[GenerationConfig] = []
        self._bind(model, options)

    def _async_client(self, config):
        return object()

    def _next_response(self) -> LLMResponse:
        self.vendor_calls += 1
        if self._errors:
            raise self._errors.pop(0)
        return make_fake_llm_response("ok")

    def _invoke_sync(self, client, prompt, config, tools):
        self.seen_configs.append(config)
        return self._next_response()

    async def _invoke_async(self, client, prompt, config, tools):
        self.seen_configs.append(config)
        return self._next_response()

    def _from_native(self, response, model):
        return response

    def _stream(self, prompt, config=None, tools=None) -> AsyncIterator[StreamChunk]:
        raise NotImplementedError


class FakeStreamingProvider(FakeProvider):
    """A provider whose stream yields one chunk, then fails with `stream_error`."""

    def __init__(self, stream_error: BaseException) -> None:
        super().__init__([])
        self._stream_error = stream_error

    async def _stream(self, prompt, config=None, tools=None):
        yield StreamChunk(delta="partial")
        raise self._stream_error


class FakeConfig(GenerationConfig):
    effort: str | None = None


class OtherProviderConfig(GenerationConfig):
    quirk: str | None = None


class FakeEffortProvider(FakeProvider):
    """A provider whose config class has a field of its own."""

    _config_cls = FakeConfig


class TestAgenerate:
    @pytest.mark.asyncio
    async def test_a_permanent_error_fails_after_one_call(self, no_retry_sleep, caplog):
        vendor_error = VendorBadRequest("400 bad request")
        provider = FakeProvider([vendor_error] * 5)

        with caplog.at_level(logging.ERROR, logger=_BASE_LOGGER):
            with pytest.raises(ProviderError) as caught:
                await provider.agenerate(prompt=None)

        assert provider.vendor_calls == 1
        assert caught.value.retryable is False
        assert caught.value.__cause__ is vendor_error
        assert caught.value.code == "provider_bad_request"
        assert str(caught.value) == "fake request failed: 400 bad request"

    @pytest.mark.asyncio
    async def test_a_transient_error_is_retried_until_it_succeeds(self, no_retry_sleep):
        provider = FakeProvider([TimeoutError(), ConnectionError()])

        result = await provider.agenerate(prompt=None)

        assert result.message.text == "ok"
        assert provider.vendor_calls == 3

    @pytest.mark.asyncio
    async def test_a_persistent_transient_error_stops_after_max_retries_plus_one(
        self, no_retry_sleep, recorded_spans
    ):
        provider = FakeProvider(
            [TimeoutError("slow")] * 10, ClientOptions(max_retries=2)
        )

        with pytest.raises(ProviderError) as caught:
            await provider.agenerate(prompt=None)

        assert provider.vendor_calls == 3
        assert caught.value.retryable is True
        assert isinstance(caught.value.__cause__, TimeoutError)
        assert [s.name for s in recorded_spans.get_finished_spans()] == ["chat"] * 3

    @pytest.mark.asyncio
    async def test_zero_retries_gives_one_call(self, no_retry_sleep):
        provider = FakeProvider([TimeoutError()] * 5, ClientOptions(max_retries=0))

        with pytest.raises(ProviderError):
            await provider.agenerate(prompt=None)

        assert provider.vendor_calls == 1

    @pytest.mark.asyncio
    async def test_the_config_max_retries_wins_over_the_client_option(
        self, no_retry_sleep
    ):
        provider = FakeProvider([TimeoutError()] * 5, ClientOptions(max_retries=5))

        with pytest.raises(ProviderError):
            await provider.agenerate(
                prompt=None, config=GenerationConfig(model="m", max_retries=1)
            )

        assert provider.vendor_calls == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "errors",
        [[VendorBadRequest("nope")], [TimeoutError()] * 4],
        ids=["permanent", "transient-exhausted"],
    )
    async def test_the_final_failure_is_logged_once_without_a_traceback(
        self, no_retry_sleep, caplog, errors
    ):
        provider = FakeProvider(errors)

        with caplog.at_level(logging.WARNING, logger=_BASE_LOGGER):
            with pytest.raises(ProviderError) as caught:
                await provider.agenerate(prompt=None)

        records = _final_failure_records(caplog)
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING
        assert records[0].exc_info is None
        assert "fake" in records[0].getMessage()
        assert caught.value.code in records[0].getMessage()

    @pytest.mark.asyncio
    async def test_the_span_records_the_vendor_error_type(self, recorded_spans):
        provider = FakeProvider([VendorBadRequest("nope")])

        with pytest.raises(ProviderError):
            await provider.agenerate(prompt=None)

        (span,) = recorded_spans.get_finished_spans()
        assert isinstance(span.error, VendorBadRequest)


class TestGenerate:
    def test_a_permanent_error_fails_after_one_call(self, no_retry_sleep, caplog):
        vendor_error = VendorBadRequest("400 bad request")
        provider = FakeProvider([vendor_error] * 5)

        with caplog.at_level(logging.WARNING, logger=_BASE_LOGGER):
            with pytest.raises(ProviderError) as caught:
                provider.generate(prompt=None)

        assert provider.vendor_calls == 1
        assert caught.value.retryable is False
        assert caught.value.__cause__ is vendor_error
        (record,) = _final_failure_records(caplog)
        assert record.levelno == logging.WARNING
        assert record.exc_info is None
        assert "fake" in record.getMessage()
        assert "provider_bad_request" in record.getMessage()

    def test_a_transient_error_is_retried_until_it_succeeds(self, no_retry_sleep):
        provider = FakeProvider([ConnectionError(), TimeoutError()])

        assert provider.generate(prompt=None).message.text == "ok"
        assert provider.vendor_calls == 3

    def test_a_persistent_transient_error_stops_after_max_retries_plus_one(
        self, no_retry_sleep, recorded_spans
    ):
        provider = FakeProvider(
            [TimeoutError("slow")] * 10, ClientOptions(max_retries=2)
        )

        with pytest.raises(ProviderError) as caught:
            provider.generate(prompt=None)

        assert provider.vendor_calls == 3
        assert caught.value.retryable is True
        assert [s.name for s in recorded_spans.get_finished_spans()] == ["chat"] * 3

    def test_zero_retries_gives_one_call(self, no_retry_sleep):
        provider = FakeProvider([TimeoutError()] * 5, ClientOptions(max_retries=0))

        with pytest.raises(ProviderError):
            provider.generate(prompt=None)

        assert provider.vendor_calls == 1


class TestTimeout:
    def test_defaults_to_120_seconds(self):
        provider = FakeProvider([])

        assert DEFAULT_LLM_TIMEOUT_SECONDS == 120.0
        assert provider._timeout(GenerationConfig()) == 120.0

    def test_the_client_option_wins_over_the_default(self):
        provider = FakeProvider([], ClientOptions(timeout=30.0))

        assert provider._timeout(GenerationConfig()) == 30.0

    def test_the_config_wins_over_the_client_option(self):
        provider = FakeProvider([], ClientOptions(timeout=30.0))

        assert provider._timeout(GenerationConfig(timeout=5.0)) == 5.0


_PROVIDER_CLASSES = [
    ("agent_platform.integrations.llm.openai.provider", "OpenAILLM"),
    ("agent_platform.integrations.llm.anthropic.provider", "AnthropicLLM"),
    ("agent_platform.integrations.llm.google.provider", "GoogleLLM"),
    ("agent_platform.integrations.llm.huggingface.provider", "HuggingFaceLLM"),
    ("agent_platform.integrations.llm.mistral.provider", "MistralLLM"),
    ("agent_platform.integrations.llm.ollama.provider", "OllamaLLM"),
]


def _load(module_name: str, class_name: str):
    import importlib

    return getattr(importlib.import_module(module_name), class_name)


def _stub_vendor(provider) -> AsyncMock:
    """Replaces the SDK client and the response parsing, so only the request
    assembly from the config runs for real."""
    client = AsyncMock()
    provider._async_client = lambda config: client
    provider._from_native = lambda response, model: make_fake_llm_response("ok")
    return client


_PROMPT = Prompt(messages=[UserMessage(content="hi")])

# The SDK client call each provider's `agenerate` makes, then its `stream`.
_CALL_PATH = {
    "OpenAILLM": "chat.completions.create",
    "AnthropicLLM": "messages.create",
    "GoogleLLM": "aio.models.generate_content",
    "HuggingFaceLLM": "chat_completion",
    "MistralLLM": "chat.complete_async",
    "OllamaLLM": "chat",
}
_STREAM_PATH = {
    **_CALL_PATH,
    "GoogleLLM": "aio.models.generate_content_stream",
    "MistralLLM": "chat.stream_async",
}
_KEY_ENV = {
    "OpenAILLM": (OpenAICredentials, "OPENAI_API_KEY"),
    "AnthropicLLM": (AnthropicCredentials, "ANTHROPIC_API_KEY"),
    "GoogleLLM": (GoogleCredentials, "GOOGLE_API_KEY"),
    "HuggingFaceLLM": (HuggingFaceCredentials, "HF_TOKEN"),
    "MistralLLM": (MistralCredentials, "MISTRAL_API_KEY"),
    "OllamaLLM": (OllamaCredentials, None),
}


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://vendor.test/v1/chat")


def _response(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status, request=_request(), headers=headers)


def _provider(class_name: str):
    module_name = next(m for m, c in _PROVIDER_CLASSES if c == class_name)
    return _load(module_name, class_name)(model="m")


def _status_error(
    class_name: str, status: int, headers: dict[str, str] | None = None
) -> Exception:
    """The error `class_name`'s SDK raises for an HTTP response with `status`."""
    message = "vendor says no"
    response = _response(status, headers)
    match class_name:
        case "OpenAILLM":
            return openai.APIStatusError(message, response=response, body=None)
        case "AnthropicLLM":
            return anthropic.APIStatusError(message, response=response, body=None)
        case "GoogleLLM":
            return genai_errors.APIError(status, {"error": {"message": message}})
        case "HuggingFaceLLM":
            return HfHubHTTPError(message, response=response)
        case "MistralLLM":
            return MistralError(message, response)
        case _:
            return ResponseError(message, status)


def _connection_error(class_name: str) -> Exception:
    if class_name == "OpenAILLM":
        return openai.APIConnectionError(request=_request())
    if class_name == "AnthropicLLM":
        return anthropic.APIConnectionError(request=_request())
    return httpx.ConnectError("refused")


def _timeout_error(class_name: str) -> Exception:
    if class_name == "OpenAILLM":
        return openai.APITimeoutError(request=_request())
    if class_name == "AnthropicLLM":
        return anthropic.APITimeoutError(request=_request())
    return httpx.ReadTimeout("slow")


def _failing_client(path: str, call: AsyncMock) -> MagicMock:
    client = MagicMock()
    target = client
    *parents, leaf = path.split(".")
    for name in parents:
        target = getattr(target, name)
    setattr(target, leaf, call)
    return client


async def _raising_events(error: Exception) -> AsyncIterator[StreamChunk]:
    raise error
    yield  # pragma: no cover - makes this an async generator


def _stub_failing_call(provider, error: Exception) -> None:
    """Makes the provider's non-stream SDK call raise `error`."""
    path = _CALL_PATH[type(provider).__name__]
    client = _failing_client(path, AsyncMock(side_effect=error))
    provider._async_client = lambda config: client


def _stub_failing_stream(provider, error: Exception, when: str = "call") -> None:
    """Makes the provider's streaming SDK call raise `error`, when it is made
    (`"call"`) or when its events are first read (`"iteration"`)."""
    path = _STREAM_PATH[type(provider).__name__]
    call = (
        AsyncMock(side_effect=error)
        if when == "call"
        else AsyncMock(return_value=_raising_events(error))
    )
    client = _failing_client(path, call)
    provider._async_client = lambda config: client
    provider._stream_client = lambda config: client


@pytest.mark.parametrize(("module_name", "class_name"), _PROVIDER_CLASSES)
class TestBaseConfigOnEveryProvider:
    @pytest.mark.asyncio
    async def test_a_base_config_with_a_model_builds_the_request(
        self, module_name, class_name
    ):
        provider = _load(module_name, class_name)()
        _stub_vendor(provider)

        result = await provider.agenerate(_PROMPT, GenerationConfig(model="m"))

        assert result.message.text == "ok"

    @pytest.mark.asyncio
    async def test_a_config_less_call_uses_the_bound_model(
        self, module_name, class_name
    ):
        provider = _load(module_name, class_name)(model="bound")
        _stub_vendor(provider)

        result = await provider.agenerate(_PROMPT)

        assert result.message.text == "ok"

    @pytest.mark.asyncio
    async def test_a_call_with_no_model_anywhere_raises_a_config_error(
        self, module_name, class_name
    ):
        provider = _load(module_name, class_name)()
        _stub_vendor(provider)

        with pytest.raises(ConfigError):
            await provider.agenerate(_PROMPT, GenerationConfig())

    @pytest.mark.asyncio
    async def test_a_stream_with_no_model_anywhere_raises_a_config_error(
        self, module_name, class_name
    ):
        provider = _load(module_name, class_name)()

        with pytest.raises(ConfigError):
            async for _ in provider.stream(_PROMPT):
                pass

    def test_an_unknown_option_key_raises_a_config_error(self, module_name, class_name):
        with pytest.raises(ConfigError, match="not_a_field"):
            _load(module_name, class_name)(options={"not_a_field": 1})

    def test_valid_options_are_accepted(self, module_name, class_name):
        _load(module_name, class_name)(options={"temperature": 0.2})

    @pytest.mark.asyncio
    @pytest.mark.parametrize("when", ["call", "iteration"])
    async def test_a_streamed_bad_request_is_a_permanent_provider_error(
        self, module_name, class_name, when
    ):
        provider = _load(module_name, class_name)(model="m")
        vendor_error = _status_error(class_name, 400)
        _stub_failing_stream(provider, vendor_error, when=when)

        with pytest.raises(ProviderError) as caught:
            async for _ in provider.stream(_PROMPT):
                pass

        assert caught.value.retryable is False
        assert caught.value.code == "provider_bad_request"
        assert caught.value.__cause__ is vendor_error

    @pytest.mark.asyncio
    @pytest.mark.parametrize("when", ["call", "iteration"])
    async def test_a_streamed_connection_error_is_retryable(
        self, module_name, class_name, when
    ):
        provider = _load(module_name, class_name)(model="m")
        _stub_failing_stream(provider, _connection_error(class_name), when=when)

        with pytest.raises(ProviderError) as caught:
            async for _ in provider.stream(_PROMPT):
                pass

        assert caught.value.retryable is True
        assert caught.value.code == "provider_unavailable"

    @pytest.mark.asyncio
    async def test_a_streamed_429_is_a_rate_limit_error(self, module_name, class_name):
        provider = _load(module_name, class_name)(model="m")
        _stub_failing_stream(provider, _status_error(class_name, 429))

        with pytest.raises(LLMRateLimitError) as caught:
            async for _ in provider.stream(_PROMPT):
                pass

        assert isinstance(caught.value, ProviderError)
        assert caught.value.code == "llm_rate_limited"
        assert caught.value.retryable is True

    @pytest.mark.asyncio
    async def test_a_429_is_a_rate_limit_error(
        self, module_name, class_name, no_retry_sleep
    ):
        provider = _load(module_name, class_name)(model="m")
        _stub_failing_call(provider, _status_error(class_name, 429))

        with pytest.raises(LLMRateLimitError) as caught:
            await provider.agenerate(_PROMPT)

        assert caught.value.code == "llm_rate_limited"
        assert caught.value.retryable is True

    @pytest.mark.asyncio
    async def test_a_503_is_a_retryable_unavailable_error(
        self, module_name, class_name, no_retry_sleep
    ):
        provider = _load(module_name, class_name)(model="m")
        _stub_failing_call(provider, _status_error(class_name, 503))

        with pytest.raises(ProviderError) as caught:
            await provider.agenerate(_PROMPT)

        assert caught.value.code == "provider_unavailable"
        assert caught.value.retryable is True

    @pytest.mark.asyncio
    async def test_a_400_is_a_permanent_provider_error(
        self, module_name, class_name, no_retry_sleep
    ):
        provider = _load(module_name, class_name)(model="m")
        _stub_failing_call(provider, _status_error(class_name, 400))

        with pytest.raises(ProviderError) as caught:
            await provider.agenerate(_PROMPT)

        assert caught.value.code == "provider_bad_request"
        assert caught.value.retryable is False

    @pytest.mark.asyncio
    async def test_a_vendor_timeout_is_a_timeout_error(
        self, module_name, class_name, no_retry_sleep
    ):
        provider = _load(module_name, class_name)(model="m")
        _stub_failing_call(provider, _timeout_error(class_name))

        with pytest.raises(LLMTimeoutError) as caught:
            await provider.agenerate(_PROMPT)

        assert caught.value.code == "llm_timeout"
        assert caught.value.retryable is True

    def test_a_missing_key_is_reported_by_its_env_var_name(
        self, module_name, class_name
    ):
        credentials_cls, env_var = _KEY_ENV[class_name]
        credentials = credentials_cls(api_key=None) if env_var else credentials_cls()

        provider = _load(module_name, class_name)(credentials)

        assert provider.missing_api_key_env() == env_var

    def test_a_present_key_reports_nothing_missing(self, module_name, class_name):
        credentials_cls, env_var = _KEY_ENV[class_name]
        credentials = (
            credentials_cls(api_key=SecretStr("k")) if env_var else credentials_cls()
        )

        provider = _load(module_name, class_name)(credentials)

        assert provider.missing_api_key_env() is None


class TestStreamErrors:
    @pytest.mark.asyncio
    async def test_an_error_after_the_first_chunk_is_translated(self):
        vendor_error = VendorBadRequest("dropped")
        provider = FakeStreamingProvider(vendor_error)
        received: list[StreamChunk] = []

        with pytest.raises(ProviderError) as caught:
            async for chunk in provider.stream(_PROMPT):
                received.append(chunk)

        assert [c.delta for c in received] == ["partial"]
        assert caught.value.code == "provider_bad_request"
        assert caught.value.__cause__ is vendor_error

    @pytest.mark.asyncio
    async def test_a_transient_error_after_the_first_chunk_is_retryable(self):
        provider = FakeStreamingProvider(ConnectionError("reset"))

        with pytest.raises(ProviderError) as caught:
            async for _ in provider.stream(_PROMPT):
                pass

        assert caught.value.retryable is True
        assert caught.value.code == "provider_unavailable"

    @pytest.mark.asyncio
    async def test_a_platform_error_passes_through_untouched(self):
        original = ConfigError("bad config")
        provider = FakeStreamingProvider(original)

        with pytest.raises(ConfigError) as caught:
            async for _ in provider.stream(_PROMPT):
                pass

        assert caught.value is original

    @pytest.mark.asyncio
    async def test_a_cancellation_is_not_translated(self):
        provider = FakeStreamingProvider(asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            async for _ in provider.stream(_PROMPT):
                pass


class TestStreamedErrorEvents:
    """A vendor error event inside a 200 stream carries its kind in the body."""

    @pytest.mark.asyncio
    async def test_an_openai_server_error_event_is_retryable(self):
        provider = OpenAILLM(model="m")
        event = openai.APIError(
            "The server had an error",
            httpx.Request("POST", "https://vendor.test"),
            body={"type": "server_error", "code": None, "message": "oops"},
        )
        _stub_failing_stream(provider, event, when="iteration")

        with pytest.raises(ProviderError) as caught:
            async for _ in provider.stream(_PROMPT):
                pass

        assert caught.value.retryable is True
        assert caught.value.code == "provider_unavailable"

    @pytest.mark.asyncio
    async def test_an_anthropic_overloaded_event_is_retryable(self):
        provider = AnthropicLLM(model="m")
        event = anthropic.APIStatusError(
            "overloaded",
            response=_response(200),
            body={"type": "error", "error": {"type": "overloaded_error"}},
        )
        _stub_failing_stream(provider, event, when="iteration")

        with pytest.raises(ProviderError) as caught:
            async for _ in provider.stream(_PROMPT):
                pass

        assert caught.value.retryable is True
        assert caught.value.code == "provider_unavailable"

    @pytest.mark.asyncio
    async def test_an_anthropic_rate_limit_event_is_a_rate_limit_error(self):
        provider = AnthropicLLM(model="m")
        event = anthropic.APIStatusError(
            "limited",
            response=_response(200),
            body={"type": "error", "error": {"type": "rate_limit_error"}},
        )
        _stub_failing_stream(provider, event, when="iteration")

        with pytest.raises(LLMRateLimitError):
            async for _ in provider.stream(_PROMPT):
                pass


class TestRetryAfterAndContextWindow:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("class_name", ["OpenAILLM", "AnthropicLLM", "MistralLLM"])
    async def test_a_429_carries_the_retry_after_header(
        self, class_name, no_retry_sleep
    ):
        provider = _provider(class_name)
        _stub_failing_call(
            provider, _status_error(class_name, 429, headers={"Retry-After": "7"})
        )

        with pytest.raises(LLMRateLimitError) as caught:
            await provider.agenerate(_PROMPT)

        assert caught.value.retry_after == 7.0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("class_name", ["OpenAILLM", "AnthropicLLM", "MistralLLM"])
    async def test_an_unparsable_retry_after_is_ignored(
        self, class_name, no_retry_sleep
    ):
        provider = _provider(class_name)
        _stub_failing_call(
            provider, _status_error(class_name, 429, headers={"Retry-After": "soon"})
        )

        with pytest.raises(LLMRateLimitError) as caught:
            await provider.agenerate(_PROMPT)

        assert caught.value.retry_after is None

    @pytest.mark.asyncio
    async def test_an_openai_context_length_400_is_a_context_window_error(self):
        provider = _provider("OpenAILLM")
        error = openai.BadRequestError(
            "maximum context length is 128000 tokens",
            response=_response(400),
            body={"code": "context_length_exceeded", "type": "invalid_request_error"},
        )
        _stub_failing_call(provider, error)

        with pytest.raises(LLMContextWindowError) as caught:
            await provider.agenerate(_PROMPT)

        assert caught.value.code == "llm_context_window"
        assert caught.value.retryable is False

    @pytest.mark.asyncio
    async def test_an_anthropic_prompt_too_long_400_is_a_context_window_error(self):
        provider = _provider("AnthropicLLM")
        message = "prompt is too long: 250000 tokens > 200000 maximum"
        error = anthropic.BadRequestError(
            message,
            response=_response(400),
            body={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": message},
            },
        )
        _stub_failing_call(provider, error)

        with pytest.raises(LLMContextWindowError) as caught:
            await provider.agenerate(_PROMPT)

        assert caught.value.code == "llm_context_window"
        assert caught.value.retryable is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "body",
        [
            '{"message": "Prompt 140000 > 128000 maximum context length",'
            ' "type": "invalid_request_prompt_too_long", "code": "3059"}',
            '{"message": "too long", "code": "3059"}',
            '{"message": "too long", "type": "invalid_request_prompt_too_long"}',
        ],
    )
    async def test_a_mistral_prompt_too_long_400_is_a_context_window_error(self, body):
        provider = _provider("MistralLLM")
        response = httpx.Response(400, request=_request(), content=body.encode())
        _stub_failing_call(provider, MistralError("bad request", response))

        with pytest.raises(LLMContextWindowError) as caught:
            await provider.agenerate(_PROMPT)

        assert caught.value.code == "llm_context_window"
        assert caught.value.retryable is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "body",
        ['{"message": "bad", "type": "invalid_request"}', "[]", "<html>oops</html>"],
    )
    async def test_another_mistral_400_is_a_plain_bad_request(self, body):
        provider = _provider("MistralLLM")
        response = httpx.Response(400, request=_request(), content=body.encode())
        _stub_failing_call(provider, MistralError("bad request", response))

        with pytest.raises(ProviderError) as caught:
            await provider.agenerate(_PROMPT)

        assert caught.value.code == "provider_bad_request"

    @pytest.mark.asyncio
    async def test_another_anthropic_400_is_a_plain_bad_request(self):
        provider = _provider("AnthropicLLM")
        error = anthropic.BadRequestError(
            "messages: field required",
            response=_response(400),
            body={"type": "error", "error": {"type": "invalid_request_error"}},
        )
        _stub_failing_call(provider, error)

        with pytest.raises(ProviderError) as caught:
            await provider.agenerate(_PROMPT)

        assert caught.value.code == "provider_bad_request"


class TestParseRetryAfter:
    def test_delay_seconds(self):
        assert parse_retry_after("7") == 7.0

    def test_an_http_date_becomes_seconds_from_now(self):
        when = datetime.now(UTC) + timedelta(seconds=90)

        seconds = parse_retry_after(format_datetime(when, usegmt=True))

        assert seconds is not None
        assert 80 <= seconds <= 90

    def test_a_past_date_waits_nothing(self):
        when = datetime.now(UTC) - timedelta(seconds=90)

        assert parse_retry_after(format_datetime(when, usegmt=True)) == 0.0

    @pytest.mark.parametrize("value", [None, "", "soon", "-3", "nan"])
    def test_an_unusable_value_gives_none(self, value):
        assert parse_retry_after(value) is None


class TestResolution:
    @pytest.mark.asyncio
    async def test_the_bound_model_and_options_apply_to_a_config_less_call(self):
        provider = FakeProvider([], model="bound", options={"temperature": 0.2})

        await provider.agenerate(prompt=None)

        (config,) = provider.seen_configs
        assert (config.model, config.temperature) == ("bound", 0.2)

    @pytest.mark.asyncio
    async def test_the_call_model_wins_over_the_bound_model(self):
        provider = FakeProvider([], model="bound")

        await provider.agenerate(prompt=None, config=GenerationConfig(model="call"))

        assert provider.seen_configs[0].model == "call"

    @pytest.mark.asyncio
    async def test_an_empty_call_model_keeps_the_bound_model(self):
        provider = FakeProvider([], model="bound")

        await provider.agenerate(prompt=None, config=GenerationConfig(model=""))

        assert provider.seen_configs[0].model == "bound"

    @pytest.mark.asyncio
    async def test_a_field_set_to_none_keeps_the_option_value(self):
        provider = FakeProvider([], options={"temperature": 0.2, "max_tokens": 50})

        await provider.agenerate(
            prompt=None, config=GenerationConfig(temperature=None, max_tokens=7)
        )

        (config,) = provider.seen_configs
        assert (config.temperature, config.max_tokens) == (0.2, 7)

    @pytest.mark.asyncio
    async def test_a_config_that_sets_nothing_keeps_the_option_values(self):
        provider = FakeProvider([], options={"temperature": 0.2})

        await provider.agenerate(prompt=None, config=GenerationConfig())

        assert provider.seen_configs[0].temperature == 0.2

    def test_the_sync_path_uses_the_same_resolution(self):
        provider = FakeProvider([], model="bound", options={"temperature": 0.2})

        provider.generate(prompt=None, config=GenerationConfig(temperature=None))

        (config,) = provider.seen_configs
        assert (config.model, config.temperature) == ("bound", 0.2)

    @pytest.mark.asyncio
    async def test_a_provider_class_config_still_gets_the_bound_model_and_options(
        self,
    ):
        provider = FakeEffortProvider([], model="bound", options={"temperature": 0.2})

        await provider.agenerate(prompt=None, config=FakeConfig(effort="high"))

        (config,) = provider.seen_configs
        assert isinstance(config, FakeConfig)
        assert (config.model, config.temperature, config.effort) == (
            "bound",
            0.2,
            "high",
        )

    @pytest.mark.asyncio
    async def test_another_provider_class_config_contributes_its_base_fields_only(
        self,
    ):
        provider = FakeEffortProvider([], model="bound")

        await provider.agenerate(
            prompt=None, config=OtherProviderConfig(seed=3, quirk="x")
        )

        (config,) = provider.seen_configs
        assert isinstance(config, FakeConfig)
        assert (config.seed, config.model, config.effort) == (3, "bound", None)

    @pytest.mark.asyncio
    async def test_no_model_anywhere_raises_a_config_error(self):
        provider = FakeProvider([], model="")

        with pytest.raises(ConfigError, match="No model set for fake"):
            await provider.agenerate(prompt=None, config=GenerationConfig())

        assert provider.vendor_calls == 0

    def test_no_model_anywhere_raises_a_config_error_on_the_sync_path(self):
        provider = FakeProvider([], model="")

        with pytest.raises(ConfigError):
            provider.generate(prompt=None)


class TestOptionsValidation:
    def test_the_model_cannot_be_an_option(self):
        with pytest.raises(ConfigError, match="'model'"):
            FakeProvider([], options={"model": "x"})

    def test_an_unknown_key_is_named(self):
        with pytest.raises(ConfigError, match="not_a_field"):
            FakeProvider([], options={"not_a_field": 1})

    def test_an_invalid_value_names_the_field_and_hides_the_value(self):
        with pytest.raises(ConfigError) as caught:
            FakeProvider([], options={"temperature": "s3cret-value"})

        assert "temperature" in str(caught.value)
        assert "s3cret-value" not in str(caught.value)
        assert caught.value.__cause__ is None
        assert caught.value.__suppress_context__ is True

    @pytest.mark.asyncio
    async def test_no_option_value_reaches_the_logs(self, caplog):
        with caplog.at_level(logging.DEBUG):
            provider = FakeProvider([], options={"stop_sequences": ["s3cret-value"]})
            await provider.agenerate(prompt=None)

        assert "s3cret-value" not in caplog.text


class TestTransientStatus:
    @pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 529])
    def test_temporary_statuses(self, status):
        assert is_transient_status(status) is True

    @pytest.mark.parametrize("status", [None, 200, 400, 401, 403, 404, 422])
    def test_permanent_statuses(self, status):
        assert is_transient_status(status) is False
