from __future__ import annotations
from agent_platform.core.interfaces.reranking.config import RerankerConfig


class HuggingFaceRerankerConfig(RerankerConfig):
    # sentence_transformers.CrossEncoder(model_name=...)
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # HuggingFaceCrossEncoder has no top-level device field; it must be
    # routed through model_kwargs={"device": ...} (see huggingface.py).
    device: str = "cpu"


# sources: https://python.langchain.com (HuggingFaceCrossEncoder reference)
