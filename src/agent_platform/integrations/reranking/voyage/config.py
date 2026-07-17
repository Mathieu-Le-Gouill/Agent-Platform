from __future__ import annotations
from agent_platform.core.interfaces.reranking.config import RerankerConfig


class VoyageRerankerConfig(RerankerConfig):
    model: str = "rerank-2"
