import logging
from collections.abc import AsyncIterator

import pytest

from agent_platform.core.credentials import ClientOptions, Credentials
from agent_platform.core.errors import ProviderError
from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.interfaces.llm.response import LLMResponse, StreamChunk
from agent_platform.integrations.llm._base import (
    DEFAULT_LLM_TIMEOUT_SECONDS,
    NativeLLMProvider,
    is_transient_status,
)
from tests.helpers import make_fake_llm_response

_BASE_LOGGER = "agent_platform.integrations.llm._base"


class VendorBadRequest(Exception):
    """Stands for a permanent vendor error such as an HTTP 400."""


class FakeProvider(
    NativeLLMProvider[GenerationConfig, object, object, LLMResponse],
):
    """A provider whose vendor calls raise the scripted errors, then succeed."""

    _provider_name = "fake"

    def __init__(
        self, errors: list[Exception], client_options: ClientOptions | None = None
    ) -> None:
        self._credentials = Credentials()
        self._client_options = client_options or ClientOptions()
        self._errors = list(errors)
        self.vendor_calls = 0

    def _default_config(self) -> GenerationConfig:
        return GenerationConfig(model="fake-model")

    def _async_client(self, config):
        return object()

    def _next_response(self) -> LLMResponse:
        self.vendor_calls += 1
        if self._errors:
            raise self._errors.pop(0)
        return make_fake_llm_response("ok")

    def _invoke_sync(self, client, prompt, config, tools):
        return self._next_response()

    async def _invoke_async(self, client, prompt, config, tools):
        return self._next_response()

    def _from_native(self, response, model):
        return response

    def stream(self, prompt, config=None, tools=None) -> AsyncIterator[StreamChunk]:
        raise NotImplementedError


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
        assert "LLM generation failed: 400 bad request" in str(caught.value)

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
    async def test_the_final_failure_is_logged_once_with_its_traceback(
        self, no_retry_sleep, caplog, errors
    ):
        provider = FakeProvider(errors)

        with caplog.at_level(logging.ERROR, logger=_BASE_LOGGER):
            with pytest.raises(ProviderError):
                await provider.agenerate(prompt=None)

        records = [r for r in caplog.records if r.name == _BASE_LOGGER]
        assert len(records) == 1
        assert records[0].levelno == logging.ERROR
        assert records[0].exc_info is not None

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

        with caplog.at_level(logging.ERROR, logger=_BASE_LOGGER):
            with pytest.raises(ProviderError) as caught:
                provider.generate(prompt=None)

        assert provider.vendor_calls == 1
        assert caught.value.retryable is False
        assert caught.value.__cause__ is vendor_error
        assert len([r for r in caplog.records if r.name == _BASE_LOGGER]) == 1

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


class TestTransientStatus:
    @pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 529])
    def test_temporary_statuses(self, status):
        assert is_transient_status(status) is True

    @pytest.mark.parametrize("status", [None, 200, 400, 401, 403, 404, 422])
    def test_permanent_statuses(self, status):
        assert is_transient_status(status) is False
