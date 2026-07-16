from __future__ import annotations

from collections import defaultdict

from agent_platform.core.interfaces.classification.response import (
    ClassificationPrediction,
    ClassificationResult,
)
from agent_platform.components.base import Component
from agent_platform.components.similarity import compute_similarity, similarity_bounds
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.enums import SimilarityMetric
from agent_platform.core.schemas.score import Score


class SimilarityInput:
    def __init__(
        self,
        chunk_vectors: list[tuple[TextChunk, list[float]]],
        label_vectors: dict[str, list[float]],
    ) -> None:
        self.chunk_vectors = chunk_vectors
        self.label_vectors = label_vectors


class SimilarityConfig:
    def __init__(
        self,
        top_k: int = 5,
        threshold: float | None = None,
        multi_label: bool = False,
        unknown_label: str = "unknown",
        metric: SimilarityMetric = SimilarityMetric.COSINE,
    ) -> None:
        self.top_k = top_k
        self.threshold = threshold
        self.multi_label = multi_label
        self.unknown_label = unknown_label
        self.metric = metric


class SimilarityScorer(Component[SimilarityInput, ClassificationResult]):
    def __init__(self, config: SimilarityConfig | None = None) -> None:
        self._config = config or SimilarityConfig()

    async def arun(self, input: SimilarityInput) -> ClassificationResult:
        label_scores: dict[str, list[float]] = defaultdict(list)
        low, high = similarity_bounds(self._config.metric)

        for chunk, chunk_vec in input.chunk_vectors:
            for label, label_vec in input.label_vectors.items():
                sim = compute_similarity(chunk_vec, label_vec, self._config.metric)
                if self._config.threshold is None or sim >= self._config.threshold:
                    label_scores[label].append(sim)

        if not label_scores:
            return ClassificationResult(
                predictions=[
                    ClassificationPrediction(
                        label=self._config.unknown_label,
                        score=Score.confidence(0.0),
                    )
                ],
            )

        result: dict[str, float] = {}
        for label, scores in label_scores.items():
            scores.sort(reverse=True)
            result[label] = sum(scores[: self._config.top_k]) / min(
                len(scores), self._config.top_k
            )

        sorted_labels = sorted(result.items(), key=lambda x: x[1], reverse=True)

        if self._config.multi_label:
            predictions = [
                ClassificationPrediction(
                    label=label, score=Score.similarity(score, low=low, high=high)
                )
                for label, score in sorted_labels
                if self._config.threshold is None or score >= self._config.threshold
            ]
            if not predictions:
                predictions.append(
                    ClassificationPrediction(
                        label=self._config.unknown_label,
                        score=Score.similarity(0.0, low=low, high=high),
                    )
                )
            return ClassificationResult(predictions=predictions)

        top_label, top_score = sorted_labels[0]
        return ClassificationResult(
            predictions=[
                ClassificationPrediction(
                    label=top_label,
                    score=Score.similarity(top_score, low=low, high=high),
                )
            ],
        )
