from __future__ import annotations

import math

from agent_platform.core.schemas.enums import SimilarityMetric

__all__ = ["compute_similarity", "similarity_bounds"]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom == 0.0:
        return 0.0
    return dot / denom


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _euclidean(a: list[float], b: list[float]) -> float:
    # inverted so a distance metric becomes a bounded (0, 1] similarity score
    distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    return 1.0 / (1.0 + distance)


def _manhattan(a: list[float], b: list[float]) -> float:
    distance = sum(abs(x - y) for x, y in zip(a, b))
    return 1.0 / (1.0 + distance)


_METRIC_FUNCS = {
    SimilarityMetric.COSINE: _cosine,
    SimilarityMetric.DOT: _dot,
    SimilarityMetric.EUCLIDEAN: _euclidean,
    SimilarityMetric.MANHATTAN: _manhattan,
}

_METRIC_BOUNDS = {
    SimilarityMetric.COSINE: (-1.0, 1.0),
    SimilarityMetric.DOT: (float("-inf"), float("inf")),
    SimilarityMetric.EUCLIDEAN: (0.0, 1.0),
    SimilarityMetric.MANHATTAN: (0.0, 1.0),
}


def compute_similarity(
    a: list[float], b: list[float], metric: SimilarityMetric = SimilarityMetric.COSINE
) -> float:
    return _METRIC_FUNCS[metric](a, b)


def similarity_bounds(metric: SimilarityMetric) -> tuple[float, float]:
    return _METRIC_BOUNDS[metric]
