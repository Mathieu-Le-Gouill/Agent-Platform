from __future__ import annotations

from importlib import import_module
from typing import Any

_PROVIDERS: dict[str, str] = {
    "DeepLTranslator": "agent_platform.integrations.translation.deepl.deepl",
    "GoogleTranslator": "agent_platform.integrations.translation.google_translate.google_translate",
    "AzureTranslator": "agent_platform.integrations.translation.azure.azure",
}


def __getattr__(name: str) -> Any:
    if name in _PROVIDERS:
        return getattr(import_module(_PROVIDERS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_PROVIDERS.keys()))
