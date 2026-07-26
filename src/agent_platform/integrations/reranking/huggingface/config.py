from __future__ import annotations

from agent_platform.core.interfaces.reranking.config import RerankerConfig


class HuggingFaceRerankerConfig(RerankerConfig):
    # sentence_transformers.CrossEncoder(model_name=...)
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Forwarded directly to CrossEncoder(device=...).
    device: str = "cpu"


# sources: https://sbert.net/docs/cross_encoder/usage/usage.html
