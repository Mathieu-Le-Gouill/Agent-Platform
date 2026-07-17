from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

pytest.importorskip("deepl")
pytest.importorskip("googletrans")

from agent_platform.core.interfaces.translation.base import BaseTranslator
from agent_platform.integrations.translation.deepl.config import DeepLConfig
from agent_platform.integrations.translation.google_translate.config import (
    GoogleTranslateConfig,
)
from agent_platform.integrations.translation.providers.deepl import (
    DeepLTranslator,
    _DEEPL_TARGETS,
)
from agent_platform.integrations.translation.providers.google_translate import (
    GoogleTranslator,
    _GOOGLE_TARGETS,
)
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.enums import Language


class TestDeepLTranslator:
    def test_constructor_stores_auth_key(self):
        translator = DeepLTranslator(DeepLConfig(auth_key="test-key-123"))
        assert translator._client is not None

    def test_is_translator(self):
        translator = DeepLTranslator(DeepLConfig(auth_key="key"))
        assert isinstance(translator, BaseTranslator)

    def test_deepl_targets_contains_english(self):
        assert Language.EN in _DEEPL_TARGETS
        assert _DEEPL_TARGETS[Language.EN] == "EN-US"

    async def test_translate_flow(self):
        content = TextChunk(text="Bonjour le monde", metadata={"idx": 1})
        mock_result = MagicMock(spec=["text", "detected_source_lang"])
        mock_result.text = "Hello world"
        mock_result.detected_source_lang = "FR"

        with patch(
            "agent_platform.integrations.translation.providers.deepl.asyncio.to_thread",
            return_value=[mock_result],
        ) as mock_to_thread:
            translator = DeepLTranslator(DeepLConfig(auth_key="test-key"))
            result = await translator.translate(content, target=Language.EN)

        assert result.text == "Hello world"
        assert result.metadata["translation_provider"] == "deepl"
        assert result.metadata["detected_source_lang"] == "FR"
        assert result.metadata["idx"] == 1

        mock_to_thread.assert_called_once()
        args, kwargs = mock_to_thread.call_args
        assert kwargs["target_lang"] == "EN-US"

    async def test_translate_with_source_lang(self):
        content = TextChunk(text="Hello", metadata={})
        mock_result = MagicMock(spec=["text", "detected_source_lang"])
        mock_result.text = "Bonjour"
        mock_result.detected_source_lang = "EN"

        with patch(
            "agent_platform.integrations.translation.providers.deepl.asyncio.to_thread",
            return_value=mock_result,
        ) as mock_to_thread:
            translator = DeepLTranslator(DeepLConfig(auth_key="test-key"))
            await translator.translate(content, target=Language.FR, source=Language.EN)

        mock_to_thread.assert_called_once()
        args, kwargs = mock_to_thread.call_args
        assert kwargs["source_lang"] == "EN-US"
        assert kwargs["target_lang"] == "fr"


class TestGoogleTranslator:
    def test_constructor_no_args(self):
        translator = GoogleTranslator(GoogleTranslateConfig())
        assert translator._client is not None

    def test_is_translator(self):
        translator = GoogleTranslator(GoogleTranslateConfig())
        assert isinstance(translator, BaseTranslator)

    def test_google_targets_contains_chinese(self):
        assert Language.CH in _GOOGLE_TARGETS
        assert _GOOGLE_TARGETS[Language.CH] == "zh-cn"

    async def test_translate_flow(self):
        content = TextChunk(text="Hello", metadata={"src": "test"})
        mock_result = MagicMock(spec=["text", "src"])
        mock_result.text = "Bonjour"
        mock_result.src = "en"

        translator = GoogleTranslator(GoogleTranslateConfig())
        with patch.object(
            translator._client, "translate", new=AsyncMock(return_value=mock_result)
        ) as mock_translate:
            result = await translator.translate(content, target=Language.FR)

        assert result.text == "Bonjour"
        assert result.metadata["translation_provider"] == "google"
        assert result.metadata["detected_source_lang"] == "en"
        assert result.metadata["src"] == "test"
        mock_translate.assert_called_once_with("Hello", dest="fr", src="")
