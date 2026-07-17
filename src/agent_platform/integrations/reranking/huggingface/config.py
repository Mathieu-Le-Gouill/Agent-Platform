from __future__ import annotations
from agent_platform.core.interfaces.reranking.config import RerankerConfig


class HuggingFaceRerankerConfig(RerankerConfig):
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cpu"
