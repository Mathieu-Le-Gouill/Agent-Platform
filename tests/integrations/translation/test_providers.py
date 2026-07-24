from unittest.mock import MagicMock, patch

import deepl
import pytest

from agent_platform.core.errors import MissingCredentialError, ProviderError
from agent_platform.core.interfaces.translation.base import BaseTranslator
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.enums import Language
from agent_platform.integrations.credentials import (
    DeepLCredentials,
    GoogleTranslateCredentials,
)
from agent_platform.integrations.translation.deepl.config import DeepLConfig
from agent_platform.integrations.translation.deepl.deepl import (
    _DEEPL_TARGETS,
    DeepLTranslator,
)
from agent_platform.integrations.translation.google_translate.config import (
    GoogleTranslateConfig,
)
from agent_platform.integrations.translation.google_translate.google_translate import (
    _GOOGLE_TARGETS,
    GoogleTranslator,
)


def _deepl_credentials(auth_key: str | None = "test-key") -> DeepLCredentials:
    return DeepLCredentials(auth_key=auth_key)


class TestDeepLTranslator:
    def test_is_translator(self):
        translator = DeepLTranslator(_deepl_credentials())
        assert isinstance(translator, BaseTranslator)

    def test_deepl_targets_contains_english(self):
        assert Language.EN in _DEEPL_TARGETS
        assert _DEEPL_TARGETS[Language.EN] == "EN-US"

    async def test_missing_credentials_raises(self):
        translator = DeepLTranslator(_deepl_credentials(auth_key=None))
        content = TextChunk(text="Hello", metadata={})

        with pytest.raises(MissingCredentialError):
            await translator.translate(content, target=Language.FR)

    async def test_translate_flow(self):
        content = TextChunk(text="Bonjour le monde", metadata={"idx": 1})
        mock_result = MagicMock(spec=["text", "detected_source_lang"])
        mock_result.text = "Hello world"
        mock_result.detected_source_lang = "FR"

        with patch(
            "agent_platform.integrations.translation.deepl.deepl.deepl.Translator"
        ) as mock_translator_cls:
            mock_client = MagicMock()
            mock_client.translate_text.return_value = [mock_result]
            mock_translator_cls.return_value = mock_client

            translator = DeepLTranslator(_deepl_credentials())
            result = await translator.translate(content, target=Language.EN)

        assert result.text == "Hello world"
        assert result.metadata["translation_provider"] == "deepl"
        assert result.metadata["detected_source_lang"] == "FR"
        assert result.metadata["idx"] == 1

        mock_client.translate_text.assert_called_once()
        _, kwargs = mock_client.translate_text.call_args
        assert kwargs["target_lang"] == "EN-US"
        assert "formality" not in kwargs

    async def test_client_is_cached_across_calls(self):
        content = TextChunk(text="Hi", metadata={})
        mock_result = MagicMock(spec=["text", "detected_source_lang"])
        mock_result.text = "Salut"
        mock_result.detected_source_lang = "EN"

        with patch(
            "agent_platform.integrations.translation.deepl.deepl.deepl.Translator"
        ) as mock_translator_cls:
            mock_client = MagicMock()
            mock_client.translate_text.return_value = [mock_result]
            mock_translator_cls.return_value = mock_client

            translator = DeepLTranslator(_deepl_credentials())
            await translator.translate(content, target=Language.FR)
            await translator.translate(content, target=Language.FR)

        mock_translator_cls.assert_called_once()
        assert mock_client.translate_text.call_count == 2

    async def test_config_fields_threaded_through(self):
        content = TextChunk(text="Hi", metadata={})
        mock_result = MagicMock(spec=["text", "detected_source_lang"])
        mock_result.text = "Salut"
        mock_result.detected_source_lang = "EN"

        config = DeepLConfig(
            formality="more",
            preserve_formatting=True,
            context="a greeting",
            model_type="quality_optimized",
            glossary_id="glossary-123",
            split_sentences="nonewlines",
            tag_handling="xml",
        )

        with patch(
            "agent_platform.integrations.translation.deepl.deepl.deepl.Translator"
        ) as mock_translator_cls:
            mock_client = MagicMock()
            mock_client.translate_text.return_value = [mock_result]
            mock_translator_cls.return_value = mock_client

            translator = DeepLTranslator(_deepl_credentials())
            await translator.translate(content, target=Language.FR, config=config)

        _, kwargs = mock_client.translate_text.call_args
        assert kwargs["formality"] == "more"
        assert kwargs["preserve_formatting"] is True
        assert kwargs["context"] == "a greeting"
        assert kwargs["model_type"] == "quality_optimized"
        assert kwargs["glossary"] == "glossary-123"
        assert kwargs["split_sentences"] == "nonewlines"
        assert kwargs["tag_handling"] == "xml"

    async def test_sdk_error_is_translated_to_provider_error(self):
        content = TextChunk(text="Hi", metadata={})

        with patch(
            "agent_platform.integrations.translation.deepl.deepl.deepl.Translator"
        ) as mock_translator_cls:
            mock_client = MagicMock()
            mock_client.translate_text.side_effect = deepl.DeepLException("boom")
            mock_translator_cls.return_value = mock_client

            translator = DeepLTranslator(_deepl_credentials())

            with pytest.raises(ProviderError, match="boom"):
                await translator.translate(content, target=Language.FR)


class TestGoogleTranslator:
    def test_constructor_no_args(self):
        translator = GoogleTranslator(GoogleTranslateCredentials())
        assert isinstance(translator, BaseTranslator)

    def test_google_targets_contains_chinese(self):
        assert Language.CH in _GOOGLE_TARGETS
        assert _GOOGLE_TARGETS[Language.CH] == "zh-CN"

    async def test_translate_passes_explicit_text_format(self):
        content = TextChunk(text="Hello & goodbye", metadata={"src": "test"})

        with patch(
            "agent_platform.integrations.translation.google_translate.google_translate.google_translate"
        ) as mock_module:
            mock_client = MagicMock()
            mock_client.translate.return_value = {
                "translatedText": "Bonjour & au revoir",
                "detectedSourceLanguage": "en",
            }
            mock_module.Client.return_value = mock_client

            translator = GoogleTranslator(GoogleTranslateCredentials())
            result = await translator.translate(content, target=Language.FR)

        assert result.text == "Bonjour & au revoir"
        mock_client.translate.assert_called_once()
        _, kwargs = mock_client.translate.call_args
        assert kwargs["format_"] == "text"
        assert "model" not in kwargs

    async def test_translate_forwards_model_when_set(self):
        content = TextChunk(text="Hello", metadata={})
        config = GoogleTranslateConfig(model="nmt")

        with patch(
            "agent_platform.integrations.translation.google_translate.google_translate.google_translate"
        ) as mock_module:
            mock_client = MagicMock()
            mock_client.translate.return_value = {
                "translatedText": "Bonjour",
                "detectedSourceLanguage": "en",
            }
            mock_module.Client.return_value = mock_client

            translator = GoogleTranslator(GoogleTranslateCredentials())
            await translator.translate(content, target=Language.FR, config=config)

        _, kwargs = mock_client.translate.call_args
        assert kwargs["format_"] == "text"
        assert kwargs["model"] == "nmt"

    async def test_client_error_is_translated_to_provider_error(self):
        content = TextChunk(text="Hello", metadata={})

        with patch(
            "agent_platform.integrations.translation.google_translate.google_translate.google_translate"
        ) as mock_module:
            mock_client = MagicMock()
            mock_client.translate.side_effect = RuntimeError("boom")
            mock_module.Client.return_value = mock_client

            translator = GoogleTranslator(GoogleTranslateCredentials())

            with pytest.raises(ProviderError, match="boom"):
                await translator.translate(content, target=Language.FR)
