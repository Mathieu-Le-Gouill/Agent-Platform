from __future__ import annotations

from agent_platform.utils.lazy_imports import make_lazy_provider_accessors

_PROVIDERS: dict[str, str] = {
    "HuggingFaceDatasetProvider": "agent_platform.integrations.dataset.huggingface.provider",
}

PROVIDER_ALIASES: dict[str, str] = {
    "huggingface": "HuggingFaceDatasetProvider",
}

__getattr__, __dir__ = make_lazy_provider_accessors(_PROVIDERS, globals())
