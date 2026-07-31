from __future__ import annotations

from google.cloud import translate_v2 as google_translate

from agent_platform.core.credentials import (
    ClientOptions,
    resolve_client_options,
    resolve_credentials,
)
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
    # google-cloud-translate's v2 client only exposes `timeout`/`retry` as per-call
    # method kwargs, not constructor-level, so `RequestOptions.timeout`/`max_retries`
    # aren't forwarded here; that would require threading per-call kwargs through
    # `client.translate(...)` in `_invoke`, a bigger change than this task covers.
    def __init__(
        self,
        credentials: GoogleTranslateCredentials | None = None,
        client_options: ClientOptions | None = None,
    ) -> None:
        self._credentials = resolve_credentials(credentials, GoogleTranslateCredentials)
        self._client_options = resolve_client_options(client_options)

    def _default_config(self) -> GoogleTranslateConfig:
        return GoogleTranslateConfig()

    def _client(self, config: GoogleTranslateConfig) -> google_translate.Client:
        client_options: dict[str, str] = {}
        if self._client_options.base_url is not None:
            client_options["api_endpoint"] = self._client_options.base_url

        if self._credentials.credentials_path:
            return google_translate.Client.from_service_account_json(
                self._credentials.credentials_path,
                client_options=client_options or None,
            )
        return google_translate.Client(client_options=client_options or None)

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
