import pytest

from agent_platform.core.errors import ProviderError
from agent_platform.core.interfaces.translation.config import TranslationConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.enums import Language
from agent_platform.integrations.translation._base import NativeTranslator


class _FakeTranslator(NativeTranslator[TranslationConfig, object]):
    def __init__(self) -> None:
        self.invoke_calls = 0
        self.side_effect = None
        self.result = TextChunk(text="translated")

    def _default_config(self) -> TranslationConfig:
        return TranslationConfig()

    def _client(self, config: TranslationConfig) -> object:
        return object()

    def _invoke(self, client, content, target, source, config):
        self.invoke_calls += 1
        if self.side_effect is not None:
            raise self.side_effect
        return self.result


def _content() -> TextChunk:
    return TextChunk(text="hello")


class TestSync:
    def test_translate_returns_invoke_result(self):
        translator = _FakeTranslator()
        result = translator.translate(_content(), target=Language.FR)
        assert result.text == "translated"
        assert translator.invoke_calls == 1

    def test_translate_does_not_wrap_errors(self):
        translator = _FakeTranslator()
        translator.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom"):
            translator.translate(_content(), target=Language.FR)


class TestAsync:
    async def test_atranslate_delegates_to_sync(self):
        translator = _FakeTranslator()
        result = await translator.atranslate(_content(), target=Language.FR)
        assert result.text == "translated"
        assert translator.invoke_calls == 1

    async def test_retries_transient_failure_then_succeeds(
        self, no_retry_sleep, mocker
    ):
        translator = _FakeTranslator()
        calls = {"n": 0}

        def flaky(client, content, target, source, config):
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("transient")
            return TextChunk(text="ok")

        mocker.patch.object(translator, "_invoke", side_effect=flaky)
        result = await translator.atranslate(_content(), target=Language.FR)

        assert calls["n"] == 2
        assert result.text == "ok"

    async def test_translates_permanent_failure_to_provider_error(
        self, no_retry_sleep, mocker
    ):
        translator = _FakeTranslator()

        def always_fails(client, content, target, source, config):
            raise ConnectionError("boom")

        mocker.patch.object(translator, "_invoke", side_effect=always_fails)

        with pytest.raises(ProviderError, match="Translation failed"):
            await translator.atranslate(_content(), target=Language.FR)
