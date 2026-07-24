from __future__ import annotations

from importlib import import_module
from typing import Any

_PROVIDERS: dict[str, str] = {
    "OpenAIEmbeddingProvider": "agent_platform.integrations.embeddings.openai.openai",
    "MistralEmbeddingProvider": "agent_platform.integrations.embeddings.mistral.mistral",
    "OllamaEmbeddingProvider": "agent_platform.integrations.embeddings.ollama.ollama",
    "HuggingFaceEmbeddingProvider": "agent_platform.integrations.embeddings.huggingface.huggingface",
}


def __getattr__(name: str) -> Any:
    if name in _PROVIDERS:
        return getattr(import_module(_PROVIDERS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_PROVIDERS.keys()))
