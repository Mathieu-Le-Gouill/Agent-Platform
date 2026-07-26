from __future__ import annotations

from google.cloud import translate_v2 as google_translate  # type: ignore[attr-defined]

from agent_platform.core.credentials import resolve_credentials
from agent_platform.core.errors import ProviderError
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.enums import Language
from agent_platform.integrations.credentials import GoogleTranslateCredentials
from agent_platform.integrations.translation._base import NativeTranslator
from agent_platform.integrations.translation.google_translate.config import (
    GoogleTranslateConfig,
)

_GOOGLE_TARGETS: dict[Language, str] = {
    Language.CH: "zh-CN",
}


class GoogleTranslator(
    NativeTranslator[GoogleTranslateConfig, google_translate.Client]
):
    def __init__(self, credentials: GoogleTranslateCredentials | None = None) -> None:
        self._credentials = resolve_credentials(credentials, GoogleTranslateCredentials)

    def _default_config(self) -> GoogleTranslateConfig:
        return GoogleTranslateConfig()

    def _client(self, config: GoogleTranslateConfig) -> google_translate.Client:
        if self._credentials.credentials_path:
            return google_translate.Client.from_service_account_json(
                self._credentials.credentials_path
            )
        return google_translate.Client()

    def _invoke(
        self,
        client: google_translate.Client,
        content: TextChunk,
        target: Language,
        source: Language | None,
        config: GoogleTranslateConfig,
    ) -> TextChunk:
        target_lang = _GOOGLE_TARGETS.get(target, target.value)
        source_lang = _GOOGLE_TARGETS.get(source, source.value) if source else None

        extra_kwargs: dict[str, object] = {}
        if config.model is not None:
            extra_kwargs["model"] = config.model

        try:
            result = client.translate(
                content.text,
                target_language=target_lang,
                source_language=source_lang,
                format_=config.format,
                **extra_kwargs,
            )
        except Exception as exc:
            raise ProviderError(f"Google Translate failed: {exc}") from exc

        return TextChunk(
            text=result["translatedText"],
            metadata={
                **content.metadata,
                "translation_provider": "google",
                "detected_source_lang": result.get("detectedSourceLanguage"),
            },
        )
