from __future__ import annotations

from agent_platform.utils.lazy_imports import make_lazy_provider_accessors

_PROVIDERS: dict[str, str] = {
    "DeepLTranslator": "agent_platform.integrations.translation.deepl.deepl",
    "GoogleTranslator": "agent_platform.integrations.translation.google_translate.google_translate",
    "AzureTranslator": "agent_platform.integrations.translation.azure.azure",
}

__getattr__, __dir__ = make_lazy_provider_accessors(_PROVIDERS, globals())
