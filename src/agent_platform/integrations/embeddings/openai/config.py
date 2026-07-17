from __future__ import annotations
from agent_platform.core.interfaces.embeddings.config import EmbeddingConfig


class OpenAIEmbeddingConfig(EmbeddingConfig):
    model: str = "text-embedding-ada-002"
    max_retries: int | None = None
    model_kwargs: dict | None = None
