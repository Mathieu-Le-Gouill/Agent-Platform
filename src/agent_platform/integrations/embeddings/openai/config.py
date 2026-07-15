from __future__ import annotations
from pydantic import BaseModel, ConfigDict
from agent_platform.core.interfaces.embeddings.config import EmbeddingConfig


class OpenAIEmbeddingConfig(EmbeddingConfig):
    model_config = ConfigDict(populate_by_name=True, frozen=True)
    model: str = "text-embedding-ada-002"
    max_retries: int | None = None
    model_kwargs: dict | None = None
