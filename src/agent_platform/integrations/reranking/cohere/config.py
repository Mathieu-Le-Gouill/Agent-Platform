from __future__ import annotations

from agent_platform.core.interfaces.reranking.config import RerankerConfig


class CohereRerankerConfig(RerankerConfig):
    # Cohere rerank model id, e.g. rerank-v4.0-fast (docs.cohere.com/reference/rerank).
    # rerank-v3.5 is deprecated and auto-routes to rerank-4-fast as of 2026-08-01.
    model: str = "rerank-v4.0-fast"
    # Caps how many chunks a long document is split into before scoring;
    # None uses Cohere's own default.
    max_chunks_per_doc: int | None = None


# sources: https://docs.cohere.com/reference/rerank
