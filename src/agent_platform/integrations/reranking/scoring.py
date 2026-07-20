from __future__ import annotations

from agent_platform.core.schemas.score import Score, ScoreKind

__all__ = ["build_relevance_scores"]


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
