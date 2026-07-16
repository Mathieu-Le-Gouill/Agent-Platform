from __future__ import annotations
from agent_platform.core.interfaces.vector_store.config import VectorStoreConfig


class PineconeConfig(VectorStoreConfig):
    environment: str = "gcp-starter"
