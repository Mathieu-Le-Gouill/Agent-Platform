from __future__ import annotations

from importlib import import_module
from typing import Any

_PROVIDERS: dict[str, str] = {
    "ChromaStore": "agent_platform.integrations.vector_store.chroma.chroma",
    "FAISSStore": "agent_platform.integrations.vector_store.faiss.faiss",
    "PineconeStore": "agent_platform.integrations.vector_store.pinecone.pinecone",
    "QdrantVectorStoreProvider": "agent_platform.integrations.vector_store.qdrant.qdrant",
    "WeaviateStore": "agent_platform.integrations.vector_store.weaviate.weaviate",
}


def __getattr__(name: str) -> Any:
    if name in _PROVIDERS:
        return getattr(import_module(_PROVIDERS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_PROVIDERS.keys()))
