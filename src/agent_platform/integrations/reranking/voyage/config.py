from __future__ import annotations
from agent_platform.core.interfaces.reranking.config import RerankerConfig


class VoyageRerankerConfig(RerankerConfig):
    # rerank-2/rerank-2-lite are superseded by rerank-2.5/rerank-2.5-lite
    # (same price, longer context, instruction-following).
    model: str = "rerank-2.5"
    # Whether to truncate inputs exceeding the model's context length instead of erroring.
    truncation: bool | None = None

    # NOTE: langchain_voyageai.VoyageAIRerank (installed version) has no
    # return_documents field -- compress_documents always returns full
    # Document objects. Library gap: not exposed as a config field.


"""
sources: https://docs.voyageai.com/docs/reranker
         https://docs.voyageai.com/reference/reranker-api
         https://blog.voyageai.com/2025/08/11/rerank-2-5
"""
