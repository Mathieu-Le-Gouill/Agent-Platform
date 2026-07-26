from __future__ import annotations

from agent_platform.core.interfaces.reranking.config import RerankerConfig


class CohereRerankerConfig(RerankerConfig):
    # Cohere rerank model id, e.g. rerank-v4.0-fast (docs.cohere.com/reference/rerank).
    # rerank-v3.5 is deprecated and auto-routes to rerank-4-fast as of 2026-08-01.
    model: str = "rerank-v4.0-fast"

    # NOTE: langchain_cohere.CohereRerank.rerank() accepts a
    # max_tokens_per_doc kwarg, but it's a per-call argument on .rerank(),
    # not a constructor field, and compress_documents() (which we use via
    # BaseDocumentCompressor.acompress_documents) never forwards it — the
    # langchain wrapper always falls back to its own 4000-token default.
    # Library gap: not exposed as a config field until the wrapper forwards
    # it through compress_documents/acompress_documents.


# sources: https://docs.cohere.com/reference/rerank
