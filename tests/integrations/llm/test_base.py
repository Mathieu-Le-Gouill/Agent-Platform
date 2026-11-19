import logging
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest

from agent_platform.core.credentials import ClientOptions, Credentials
from agent_platform.core.errors import ConfigError, ProviderError
from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.interfaces.llm.response import LLMResponse, StreamChunk
from agent_platform.core.schemas.message import Prompt, UserMessage
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

    def stream(self, prompt, config=None, tools=None) -> AsyncIterator[StreamChunk]:
        raise NotImplementedError


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
