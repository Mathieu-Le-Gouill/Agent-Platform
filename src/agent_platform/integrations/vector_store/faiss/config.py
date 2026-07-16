from __future__ import annotations
from agent_platform.core.interfaces.vector_store.config import VectorStoreConfig


class FAISSConfig(VectorStoreConfig):
    index_path: str | None = None
