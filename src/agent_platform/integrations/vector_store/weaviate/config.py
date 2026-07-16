from __future__ import annotations
from pydantic import ConfigDict
from agent_platform.core.interfaces.vector_store.config import VectorStoreConfig


class WeaviateConfig(VectorStoreConfig):
    model_config = ConfigDict(populate_by_name=True, frozen=True)
    text_key: str = "text"
