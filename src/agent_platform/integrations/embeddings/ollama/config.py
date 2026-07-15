from __future__ import annotations
from pydantic import BaseModel, ConfigDict
from agent_platform.core.interfaces.embeddings.config import EmbeddingConfig


class OllamaEmbeddingConfig(EmbeddingConfig):
    model_config = ConfigDict(populate_by_name=True, frozen=True)
    model: str = "nomic-embed-text"
    top_p: float | None = None
    top_k: int | None = None
    temperature: float | None = None
