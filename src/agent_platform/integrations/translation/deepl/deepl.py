from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import deepl

if TYPE_CHECKING:
    from deepl import TextResult

from agent_platform.integrations.credentials import DeepLCredentials
from agent_platform.core.interfaces.translation.base import BaseTranslator
from agent_platform.integrations.translation.deepl.config import DeepLConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.enums import Language
from agent_platform.core.errors import ProviderError, error_logged, with_retry


_DEEPL_TARGETS: dict[Language, str] = {
    Language.EN: "EN-US",
}


class DeepLTranslator(BaseTranslator[DeepLConfig]):
    def __init__(self, credentials: DeepLCredentials | None = None) -> None:
        self._credentials = (
            credentials if credentials is not None else DeepLCredentials()
        )

    def _default_config(self) -> DeepLConfig:
        return DeepLConfig()

    @error_logged(re_raise=ProviderError, message="Translation failed")
    @with_retry()
    async def translate(
        self,
        content: TextChunk,
        target: Language,
        source: Language | None = None,
        config: DeepLConfig | None = None,
    ) -> TextChunk:
        config = config or self._default_config()
        client = deepl.Translator(self._credentials.auth_key.get_secret_value())

        target_lang = _DEEPL_TARGETS.get(target, target.value)
        source_lang = _DEEPL_TARGETS.get(source) if source else None

        raw = await asyncio.to_thread(
            client.translate_text,
            content.text,
            target_lang=target_lang,
            source_lang=source_lang,
        )

        result: TextResult = raw[0] if isinstance(raw, list) else raw

        return TextChunk(
            text=result.text,
            metadata={
                **content.metadata,
                "translation_provider": "deepl",
                "detected_source_lang": result.detected_source_lang,
            },
        )
