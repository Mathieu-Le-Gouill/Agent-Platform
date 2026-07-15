from __future__ import annotations
from pydantic import BaseModel, ConfigDict
from agent_platform.core.interfaces.embeddings.config import EmbeddingConfig


class MistralEmbeddingConfig(EmbeddingConfig):
    model_config = ConfigDict(populate_by_name=True, frozen=True)
    model: str = "mistral-embed"
    max_retries: int | None = None
    endpoint: str = "https://api.mistral.ai/v1/"
    wait_time: int | None = None
    max_concurrent_requests: int | None = None
