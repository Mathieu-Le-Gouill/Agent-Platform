from __future__ import annotations
from pydantic import ConfigDict
from agent_platform.core.interfaces.vector_store.config import VectorStoreConfig


class FAISSConfig(VectorStoreConfig):
    model_config = ConfigDict(populate_by_name=True, frozen=True)
    index_path: str | None = None
