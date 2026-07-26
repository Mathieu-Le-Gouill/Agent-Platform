from __future__ import annotations

from typing import TYPE_CHECKING, Any

import deepl

if TYPE_CHECKING:
    from deepl import TextResult

from agent_platform.core.credentials import resolve_credentials
from agent_platform.core.errors import ProviderError, require_secret
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.enums import Language
from agent_platform.integrations.credentials import DeepLCredentials
from agent_platform.integrations.translation._base import NativeTranslator
from agent_platform.integrations.translation.deepl.config import DeepLConfig

_DEEPL_TARGETS: dict[Language, str] = {
    Language.EN: "EN-US",
}


class DeepLTranslator(NativeTranslator[DeepLConfig, deepl.Translator]):
    def __init__(self, credentials: DeepLCredentials | None = None) -> None:
        self._credentials = resolve_credentials(credentials, DeepLCredentials)
        self._client_cache: deepl.Translator | None = None

    def _default_config(self) -> DeepLConfig:
        return DeepLConfig()

    def _client(self, config: DeepLConfig) -> deepl.Translator:
        if self._client_cache is None:
            auth_key = require_secret(
                self._credentials.auth_key, "DeepL auth key is required"
            )
            try:
                self._client_cache = deepl.Translator(auth_key.get_secret_value())
            except deepl.DeepLException as exc:
                raise ProviderError(
                    f"DeepL client initialization failed: {exc}"
                ) from exc
        return self._client_cache

    def _invoke(
        self,
        client: deepl.Translator,
        content: TextChunk,
        target: Language,
        source: Language | None,
        config: DeepLConfig,
    ) -> TextChunk:
        target_lang = _DEEPL_TARGETS.get(target, target.value)
        source_lang = _DEEPL_TARGETS.get(source) if source else None

        extra_kwargs: dict[str, Any] = {}
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
            raw = client.translate_text(
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
