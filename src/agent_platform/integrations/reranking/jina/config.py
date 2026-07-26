from __future__ import annotations

from agent_platform.core.interfaces.reranking.config import RerankerConfig


class JinaRerankerConfig(RerankerConfig):
    # Jina reranker model id, e.g. jina-reranker-v3 (jina.ai/reranker).
    model: str = "jina-reranker-v3"


# sources: https://jina.ai/reranker/
