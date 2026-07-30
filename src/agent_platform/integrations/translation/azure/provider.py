from __future__ import annotations

from azure.ai.translation.text import TextTranslationClient
from azure.ai.translation.text.models import TranslateInputItem, TranslationTarget
from azure.core.credentials import AzureKeyCredential

from agent_platform.core.credentials import (
    ClientOptions,
    resolve_client_options,
    resolve_credentials,
    resolve_max_retries,
    resolve_timeout,
)
from agent_platform.core.errors import ProviderError, require_secret
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.enums import Language
from agent_platform.integrations.credentials import AzureTranslatorCredentials
from agent_platform.integrations.translation._base import NativeTranslator
from agent_platform.integrations.translation.azure.config import AzureTranslatorConfig

_DEFAULT_ENDPOINT = "https://api.cognitive.microsofttranslator.com"


class AzureTranslator(NativeTranslator[AzureTranslatorConfig, TextTranslationClient]):
    def __init__(
        self,
        credentials: AzureTranslatorCredentials | None = None,
        client_options: ClientOptions | None = None,
    ) -> None:
        self._credentials = resolve_credentials(credentials, AzureTranslatorCredentials)
        self._client_options = resolve_client_options(client_options)

    def _default_config(self) -> AzureTranslatorConfig:
        return AzureTranslatorConfig()

    def _client(self, config: AzureTranslatorConfig) -> TextTranslationClient:
        api_key = require_secret(
            self._credentials.api_key, "Azure Translator API key is required"
        )
        credential = AzureKeyCredential(api_key.get_secret_value())
        kwargs: dict[str, int | float] = {
            "retry_total": resolve_max_retries(
                config.max_retries, self._client_options
            ),
        }
        timeout = resolve_timeout(config.timeout, self._client_options)
        if timeout is not None:
            kwargs["connection_timeout"] = timeout
            kwargs["read_timeout"] = timeout
        return TextTranslationClient(
            endpoint=self._client_options.base_url or _DEFAULT_ENDPOINT,
            credential=credential,
            region=self._credentials.region,
            api_version=config.api_version,
            **kwargs,  # type: ignore[arg-type]
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
