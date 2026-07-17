from __future__ import annotations
from agent_platform.core.interfaces.reranking.config import RerankerConfig


class CohereRerankerConfig(RerankerConfig):
    model: str = "rerank-v3.5"
