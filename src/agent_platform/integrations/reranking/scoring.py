from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

from agent_platform.core.interfaces.reranking.config import RerankerConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import Score, ScoreKind

__all__ = ["apply_rerank_results", "build_relevance_scores"]

ResultT = TypeVar("ResultT")


def _min_max_normalize(raw: list[float | None]) -> list[float | None]:
    values = [v for v in raw if v is not None]
    if not values:
        return raw
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0 if v is not None else None for v in raw]
    return [None if v is None else (v - lo) / (hi - lo) for v in raw]


def build_relevance_scores(
    raw: list[float | None], *, normalize: bool
) -> list[Score | None]:
    values = _min_max_normalize(raw) if normalize else raw
    scores: list[Score | None] = []
    for value in values:
        if value is None:
            scores.append(None)
        elif normalize:
            scores.append(Score.relevance(value))
        else:
            # Raw provider scores aren't guaranteed to fall inside [0, 1]
            # (e.g. HuggingFace cross-encoder logits), so use unbounded
            # low/high to avoid a spurious Score validation error.
            scores.append(Score.logit(value, kind=ScoreKind.RELEVANCE))
    return scores


def apply_rerank_results(
    items: Sequence[TextChunk],
    results: Sequence[ResultT],
    config: RerankerConfig,
    *,
    index: Callable[[ResultT], int],
    score: Callable[[ResultT], float | None],
) -> list[TextChunk]:
    """Map vendor-neutral rerank results (already in relevance order) back onto
    `items`, attaching a `confidence` score when requested, then re-slicing to
    `config.top_k`. Shared by every reranking provider regardless of whether it
    calls a native SDK, a REST endpoint, or a local model."""
    scores: list[Score | None]
    if config.return_scores:
        scores = build_relevance_scores(
            [score(r) for r in results], normalize=config.normalize_scores
        )
    else:
        scores = [None] * len(results)

    reranked = [
        items[index(r)].model_copy(update={"confidence": s})
        for r, s in zip(results, scores)
    ]
    if config.top_k is not None:
        reranked = reranked[: config.top_k]
    return reranked
