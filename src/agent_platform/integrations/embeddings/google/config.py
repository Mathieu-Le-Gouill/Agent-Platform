from __future__ import annotations

from agent_platform.core.interfaces.embeddings.config import EmbeddingConfig


class GoogleEmbeddingConfig(EmbeddingConfig):
    model: str = "gemini-embedding-001"
