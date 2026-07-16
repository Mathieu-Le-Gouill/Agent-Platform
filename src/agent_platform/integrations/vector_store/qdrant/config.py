from __future__ import annotations
from pydantic import BaseModel, ConfigDict
from agent_platform.core.interfaces.vector_store.config import VectorStoreConfig


class QdrantConfig(VectorStoreConfig):
    model_config = ConfigDict(populate_by_name=True, frozen=True)
    url: str = "http://localhost:6333"
    api_key: str | None = None
