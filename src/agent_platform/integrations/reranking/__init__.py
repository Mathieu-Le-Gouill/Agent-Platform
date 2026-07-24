from __future__ import annotations

from importlib import import_module
from typing import Any

_PROVIDERS: dict[str, str] = {
    "CohereRerankerProvider": "agent_platform.integrations.reranking.cohere.cohere",
    "JinaRerankerProvider": "agent_platform.integrations.reranking.jina.jina",
    "HuggingFaceRerankerProvider": "agent_platform.integrations.reranking.huggingface.huggingface",
    "FlashRankReranker": "agent_platform.integrations.reranking.flashrank.flashrank",
    "VoyageRerankerProvider": "agent_platform.integrations.reranking.voyage.voyage",
}


def __getattr__(name: str) -> Any:
    if name in _PROVIDERS:
        return getattr(import_module(_PROVIDERS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_PROVIDERS.keys()))
