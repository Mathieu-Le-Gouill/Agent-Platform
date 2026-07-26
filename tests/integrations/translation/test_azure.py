from unittest.mock import MagicMock

import pytest

from agent_platform.core.errors import MissingCredentialError, ProviderError
from agent_platform.core.interfaces.translation.base import BaseTranslator
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.enums import Language
from agent_platform.integrations.credentials import AzureTranslatorCredentials
from agent_platform.integrations.translation.azure.config import (
    AzureTranslatorConfig,
)
from agent_platform.integrations.translation.azure.provider import AzureTranslator


def _azure_credentials(api_key: str | None = "test-key") -> AzureTranslatorCredentials:
    return AzureTranslatorCredentials(api_key=api_key, region="westus")


def _make_response(text: str = "Bonjour", detected: str | None = None):
    translation = MagicMock(spec=["text"])
    translation.text = text

    item = MagicMock(spec=["translations", "detected_language"])
    item.translations = [translation]
    if detected is not None:
        detected_language = MagicMock(spec=["language"])
        detected_language.language = detected
        item.detected_language = detected_language
    else:
        item.detected_language = None
    return [item]


class TestAzureTranslator:
    def test_is_translator(self):
        translator = AzureTranslator(_azure_credentials())
        assert isinstance(translator, BaseTranslator)

    def test_missing_credentials_raises(self):
        translator = AzureTranslator(_azure_credentials(api_key=None))
        content = TextChunk(text="Hello", metadata={})

        with pytest.raises(MissingCredentialError):
            translator.translate(content, target=Language.FR)

    def test_from_language_is_a_string_not_a_list(self, mocker):
        """Regression test: `TranslationTarget`/`TranslateInputItem` must receive
        plain strings for source/target languages, not single-element lists —
        the SDK's `TranslateInputItem.language` keyword is `str | None`."""
        content = TextChunk(text="Hello", metadata={})

        mock_client_cls = mocker.patch(
            "agent_platform.integrations.translation.azure.provider.TextTranslationClient"
        )
        mock_client = MagicMock()
        mock_client.translate.return_value = _make_response()
        mock_client_cls.return_value = mock_client

        translator = AzureTranslator(_azure_credentials())
        translator.translate(content, target=Language.FR, source=Language.EN)

        mock_client.translate.assert_called_once()
        _, kwargs = mock_client.translate.call_args
        body = kwargs["body"]
        assert len(body) == 1
        input_item = body[0]

        assert input_item.language == "en"
        assert not isinstance(input_item.language, list)

        target_item = input_item.targets[0]
        assert target_item.language == "fr"
        assert not isinstance(target_item.language, list)

    def test_source_none_leaves_language_unset(self, mocker):
        content = TextChunk(text="Hello", metadata={})

        mock_client_cls = mocker.patch(
            "agent_platform.integrations.translation.azure.provider.TextTranslationClient"
        )
        mock_client = MagicMock()
        mock_client.translate.return_value = _make_response()
        mock_client_cls.return_value = mock_client

        translator = AzureTranslator(_azure_credentials())
        translator.translate(content, target=Language.FR)

        _, kwargs = mock_client.translate.call_args
        input_item = kwargs["body"][0]
        assert input_item.language is None

    def test_translate_flow_and_response_mapping(self, mocker):
        content = TextChunk(text="Hello", metadata={"idx": 1})

        mock_client_cls = mocker.patch(
            "agent_platform.integrations.translation.azure.provider.TextTranslationClient"
        )
        mock_client = MagicMock()
        mock_client.translate.return_value = _make_response(
            text="Bonjour", detected="en"
        )
        mock_client_cls.return_value = mock_client

        translator = AzureTranslator(_azure_credentials())
        result = translator.translate(content, target=Language.FR)

        assert result.text == "Bonjour"
        assert result.metadata["translation_provider"] == "azure"
        assert result.metadata["detected_source_lang"] == "en"
        assert result.metadata["idx"] == 1

    def test_api_version_threaded_to_client(self, mocker):
        content = TextChunk(text="Hello", metadata={})
        config = AzureTranslatorConfig(api_version="2025-10-01-preview")

        mock_client_cls = mocker.patch(
            "agent_platform.integrations.translation.azure.provider.TextTranslationClient"
        )
        mock_client = MagicMock()
        mock_client.translate.return_value = _make_response()
        mock_client_cls.return_value = mock_client

        translator = AzureTranslator(_azure_credentials())
        translator.translate(content, target=Language.FR, config=config)

        _, kwargs = mock_client_cls.call_args
        assert kwargs["api_version"] == "2025-10-01-preview"

    def test_config_fields_threaded_through(self, mocker):
        content = TextChunk(text="Hello", metadata={})
        config = AzureTranslatorConfig(
            text_type="html",
            profanity_action="Marked",
            profanity_marker="Asterisk",
            allow_fallback=False,
        )

        mock_client_cls = mocker.patch(
            "agent_platform.integrations.translation.azure.provider.TextTranslationClient"
        )
        mock_client = MagicMock()
        mock_client.translate.return_value = _make_response()
        mock_client_cls.return_value = mock_client

        translator = AzureTranslator(_azure_credentials())
        translator.translate(content, target=Language.FR, config=config)

        _, kwargs = mock_client.translate.call_args
        input_item = kwargs["body"][0]
        assert input_item.text_type == "html"

        target_item = input_item.targets[0]
        assert target_item.profanity_action == "Marked"
        assert target_item.profanity_marker == "Asterisk"
        assert target_item.allow_fallback is False

    def test_client_error_is_translated_to_provider_error_on_sync_path(self, mocker):
        content = TextChunk(text="Hello", metadata={})

        mock_client_cls = mocker.patch(
            "agent_platform.integrations.translation.azure.provider.TextTranslationClient"
        )
        mock_client = MagicMock()
        mock_client.translate.side_effect = RuntimeError("boom")
        mock_client_cls.return_value = mock_client

        translator = AzureTranslator(_azure_credentials())

        with pytest.raises(ProviderError, match="boom"):
            translator.translate(content, target=Language.FR)

    async def test_atranslate_delegates_to_sync(self, mocker):
        content = TextChunk(text="Hello", metadata={})

        mock_client_cls = mocker.patch(
            "agent_platform.integrations.translation.azure.provider.TextTranslationClient"
        )
        mock_client = MagicMock()
        mock_client.translate.return_value = _make_response(text="Bonjour")
        mock_client_cls.return_value = mock_client

        translator = AzureTranslator(_azure_credentials())
        result = await translator.atranslate(content, target=Language.FR)

        assert result.text == "Bonjour"
