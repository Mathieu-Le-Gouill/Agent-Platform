from __future__ import annotations

from typing import Literal

from agent_platform.core.interfaces.translation.config import TranslationConfig


class AzureTranslatorConfig(TranslationConfig):
    # https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/translation/azure-ai-translation-text/azure/ai/translation/text/_patch.py
    # `TextTranslationClient.__init__` takes `api_version` as a date-based string
    # ("2025-10-01-preview" / "2026-06-06"), not the legacy REST "3.0" scheme.
    api_version: str = "2026-06-06"
    # https://learn.microsoft.com/azure/ai-services/translator/text-translation/reference/v3/translate — forwarded via TranslateInputItem.text_type
    text_type: Literal["plain", "html"] | None = None
    # https://learn.microsoft.com/azure/ai-services/translator/text-translation/reference/v3/translate — Custom Translator
    # category ID. Not exposed by the installed azure-ai-translation-text 2.0.0 preview SDK
    # (TranslateInputItem/TranslationTarget have no `category` field); kept for forward
    # compatibility and API documentation, currently not forwarded.
    category: str | None = None
    # https://learn.microsoft.com/azure/ai-services/translator/text-translation/reference/v3/translate — forwarded via TranslationTarget.profanity_action
    profanity_action: Literal["NoAction", "Marked", "Deleted"] | None = None
    # https://learn.microsoft.com/azure/ai-services/translator/text-translation/reference/v3/translate — forwarded via TranslationTarget.profanity_marker
    profanity_marker: Literal["Asterisk", "Tag"] | None = None
    # https://learn.microsoft.com/azure/ai-services/translator/text-translation/reference/v3/translate — response
    # alignment info. Not exposed by the installed azure-ai-translation-text 2.0.0 preview SDK
    # (TranslationText has no `alignment` field); kept for forward compatibility, currently a no-op.
    include_alignment: bool | None = None
    # https://learn.microsoft.com/azure/ai-services/translator/text-translation/reference/v3/translate — forwarded via TranslationTarget.allow_fallback
    allow_fallback: bool | None = None
