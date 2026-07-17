from __future__ import annotations

import asyncio

from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential

from agent_platform.integrations.credentials import (
    AzureTranslatorCredentials,
)
from agent_platform.core.interfaces.translation.base import BaseTranslator
from agent_platform.integrations.translation.azure.config import AzureTranslatorConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.enums import Language
from agent_platform.core.errors import (
    MissingCredentialError,
    ProviderError,
    error_logged,
    with_retry,
)


class AzureTranslator(BaseTranslator[AzureTranslatorConfig]):
    def __init__(self, credentials: AzureTranslatorCredentials | None = None) -> None:
        self._credentials = (
            credentials if credentials is not None else AzureTranslatorCredentials()
        )

    def _default_config(self) -> AzureTranslatorConfig:
        return AzureTranslatorConfig()

    def _build_client(self) -> TextTranslationClient:
        if not self._credentials.api_key:
            raise MissingCredentialError("Azure Translator API key is required")
        credential = AzureKeyCredential(self._credentials.api_key.get_secret_value())
        return TextTranslationClient(
            endpoint=self._credentials.endpoint,
            credential=credential,
            region=self._credentials.region,
        )

    @error_logged(re_raise=ProviderError, message="Translation failed")
    @with_retry()
    async def translate(
        self,
        content: TextChunk,
        target: Language,
        source: Language | None = None,
        config: AzureTranslatorConfig | None = None,
    ) -> TextChunk:
        config = config or self._default_config()
        client = self._build_client()

        try:
            response = await asyncio.to_thread(
                client.translate,
                body=[{"text": content.text}],
                to_language=[target.value],
                from_language=[source.value] if source else None,
            )
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
