from __future__ import annotations

from agent_platform.utils.lazy_imports import make_lazy_provider_accessors

_PROVIDERS: dict[str, str] = {
    "TransformersClassifier": "agent_platform.integrations.classification.transformers.provider",
}

__getattr__, __dir__ = make_lazy_provider_accessors(_PROVIDERS, globals())
