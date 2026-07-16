from __future__ import annotations
from agent_platform.core.interfaces.vector_store.config import VectorStoreConfig


class QdrantConfig(VectorStoreConfig):
    url: str = "http://localhost:6333"
    api_key: str | None = None
