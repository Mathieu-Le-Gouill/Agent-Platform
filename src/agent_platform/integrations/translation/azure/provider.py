from __future__ import annotations

from azure.ai.translation.text import TextTranslationClient
from azure.ai.translation.text.models import TranslateInputItem, TranslationTarget
from azure.core.credentials import AzureKeyCredential

from agent_platform.core.credentials import resolve_credentials
from agent_platform.core.errors import ProviderError, require_secret
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.enums import Language
from agent_platform.integrations.credentials import AzureTranslatorCredentials
from agent_platform.integrations.translation._base import NativeTranslator
from agent_platform.integrations.translation.azure.config import AzureTranslatorConfig


class AzureTranslator(NativeTranslator[AzureTranslatorConfig, TextTranslationClient]):
    def __init__(self, credentials: AzureTranslatorCredentials | None = None) -> None:
        self._credentials = resolve_credentials(credentials, AzureTranslatorCredentials)

    def _default_config(self) -> AzureTranslatorConfig:
        return AzureTranslatorConfig()

    def _client(self, config: AzureTranslatorConfig) -> TextTranslationClient:
        api_key = require_secret(
            self._credentials.api_key, "Azure Translator API key is required"
        )
        credential = AzureKeyCredential(api_key.get_secret_value())
        return TextTranslationClient(
            endpoint=self._credentials.endpoint,
            credential=credential,
            region=self._credentials.region,
            api_version=config.api_version,
        )

    def _invoke(
        self,
        client: TextTranslationClient,
        content: TextChunk,
        target: Language,
        source: Language | None,
        config: AzureTranslatorConfig,
    ) -> TextChunk:
        target_item = TranslationTarget(
            language=target.value,
            profanity_action=config.profanity_action,
            profanity_marker=config.profanity_marker,
            allow_fallback=config.allow_fallback,
        )
        input_item = TranslateInputItem(
            text=content.text,
            targets=[target_item],
            language=source.value if source else None,
            text_type=config.text_type,
        )

        try:
            response = client.translate(body=[input_item])
        except Exception as exc:
            raise ProviderError(f"Azure Translator failed: {exc}") from exc

        translation = response[0].translations[0]
        detected_lang = (
            response[0].detected_language.language
            if response[0].detected_language
            else None
        )

        return TextChunk(
            text=translation.text,
            metadata={
                **content.metadata,
                "translation_provider": "azure",
                "detected_source_lang": detected_lang,
            },
        )
