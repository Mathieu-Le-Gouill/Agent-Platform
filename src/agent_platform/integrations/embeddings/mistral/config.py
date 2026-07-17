from __future__ import annotations
from agent_platform.core.interfaces.embeddings.config import EmbeddingConfig


class MistralEmbeddingConfig(EmbeddingConfig):
    model: str = "mistral-embed"
    max_retries: int | None = None
    endpoint: str = "https://api.mistral.ai/v1/"
    wait_time: int | None = None
    max_concurrent_requests: int | None = None
