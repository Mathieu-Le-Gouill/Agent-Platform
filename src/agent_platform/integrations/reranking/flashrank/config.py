from __future__ import annotations
from agent_platform.core.interfaces.reranking.config import RerankerConfig


class FlashRankConfig(RerankerConfig):
    model: str = "ms-marco-MiniLM-L-12-v2"
    cache_dir: str | None = None
