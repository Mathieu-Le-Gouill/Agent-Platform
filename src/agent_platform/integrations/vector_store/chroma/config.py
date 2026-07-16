from __future__ import annotations
from pydantic import BaseModel, ConfigDict
from agent_platform.core.interfaces.vector_store.config import VectorStoreConfig


class ChromaConfig(VectorStoreConfig):
    model_config = ConfigDict(populate_by_name=True, frozen=True)
    host: str = "localhost"
    port: int = 8000
