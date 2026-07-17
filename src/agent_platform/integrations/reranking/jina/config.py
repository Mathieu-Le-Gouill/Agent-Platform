from __future__ import annotations
from agent_platform.core.interfaces.reranking.config import RerankerConfig


class JinaRerankerConfig(RerankerConfig):
    model: str = "jina-reranker-v2-base-multilingual"
