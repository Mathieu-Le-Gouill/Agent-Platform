from __future__ import annotations
from enum import Enum
from agent_platform.core.interfaces.embeddings.config import EmbeddingConfig


class HuggingFaceEmbeddingMode(str, Enum):
    LOCAL = "local"
    HOSTED = "hosted"


class HuggingFaceEmbeddingConfig(EmbeddingConfig):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    mode: HuggingFaceEmbeddingMode = HuggingFaceEmbeddingMode.LOCAL
    model_kwargs: dict | None = None
    encode_kwargs: dict | None = None
    provider: str | None = None
