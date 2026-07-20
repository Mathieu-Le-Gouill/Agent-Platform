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
from agent_platform.core.errors import (
    MissingCredentialError,
    ProviderError,
    error_logged,
    with_retry,
)


_DEEPL_TARGETS: dict[Language, str] = {
    Language.EN: "EN-US",
}


class DeepLTranslator(BaseTranslator[DeepLConfig]):
    def __init__(self, credentials: DeepLCredentials | None = None) -> None:
        self._credentials = (
            credentials if credentials is not None else DeepLCredentials()
        )
        self._client: deepl.Translator | None = None

    def _default_config(self) -> DeepLConfig:
        return DeepLConfig()

    def _get_client(self) -> deepl.Translator:
        if self._client is None:
            if self._credentials.auth_key is None:
                raise MissingCredentialError("DeepL auth key is required")
            try:
                self._client = deepl.Translator(
                    self._credentials.auth_key.get_secret_value()
                )
            except deepl.DeepLException as exc:
                raise ProviderError(
                    f"DeepL client initialization failed: {exc}"
                ) from exc
        return self._client

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
        client = self._get_client()

        target_lang = _DEEPL_TARGETS.get(target, target.value)
        source_lang = _DEEPL_TARGETS.get(source) if source else None

        extra_kwargs: dict[str, object] = {}
        if config.formality is not None:
            extra_kwargs["formality"] = config.formality
        if config.preserve_formatting is not None:
            extra_kwargs["preserve_formatting"] = config.preserve_formatting
        if config.context is not None:
            extra_kwargs["context"] = config.context
        if config.model_type is not None:
            extra_kwargs["model_type"] = config.model_type
        if config.glossary_id is not None:
            extra_kwargs["glossary"] = config.glossary_id
        if config.split_sentences is not None:
            extra_kwargs["split_sentences"] = config.split_sentences
        if config.tag_handling is not None:
            extra_kwargs["tag_handling"] = config.tag_handling

        try:
            raw = await asyncio.to_thread(
                client.translate_text,
                content.text,
                target_lang=target_lang,
                source_lang=source_lang,
                **extra_kwargs,
            )
        except deepl.DeepLException as exc:
            raise ProviderError(f"DeepL translation failed: {exc}") from exc

        result: TextResult = raw[0] if isinstance(raw, list) else raw

        return TextChunk(
            text=result.text,
            metadata={
                **content.metadata,
                "translation_provider": "deepl",
                "detected_source_lang": result.detected_source_lang,
            },
        )
