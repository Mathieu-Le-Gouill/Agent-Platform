from dataclasses import FrozenInstanceError
from uuid import uuid4

import pytest
from pydantic import ValidationError

from agent_platform.components.classification.similarity_scorer.component import (
    SimilarityConfig,
    SimilarityInput,
    SimilarityScorer,
)
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.enums import SimilarityMetric


def _chunk(text: str = "chunk") -> TextChunk:
    return TextChunk(id=uuid4(), text=text)


class TestSimilarityScorer:
    async def test_no_chunks_returns_unknown_label(self):
        scorer = SimilarityScorer()
        result = await scorer.arun(SimilarityInput(chunk_vectors=[], label_vectors={}))
        assert result.predictions[0].label == "unknown"
        assert result.predictions[0].score.value == 0.0

    async def test_single_label_best_match(self):
        scorer = SimilarityScorer()
        chunk_vectors = [(_chunk(), [1.0, 0.0])]
        label_vectors = {"cat": [1.0, 0.0], "dog": [0.0, 1.0]}
        result = await scorer.arun(
            SimilarityInput(chunk_vectors=chunk_vectors, label_vectors=label_vectors)
        )
        assert result.top.label == "cat"

    async def test_threshold_filters_out_low_similarity(self):
        scorer = SimilarityScorer()
        chunk_vectors = [(_chunk(), [1.0, 0.0])]
        label_vectors = {"dog": [0.0, 1.0]}
        result = await scorer.arun(
            SimilarityInput(
                chunk_vectors=chunk_vectors,
                label_vectors=label_vectors,
                config=SimilarityConfig(threshold=0.9),
            )
        )
        assert result.predictions[0].label == "unknown"

    async def test_multi_label_returns_all_above_threshold(self):
        scorer = SimilarityScorer()
        chunk_vectors = [(_chunk(), [1.0, 0.0])]
        label_vectors = {"cat": [1.0, 0.0], "dog": [0.0, 1.0]}
        result = await scorer.arun(
            SimilarityInput(
                chunk_vectors=chunk_vectors,
                label_vectors=label_vectors,
                config=SimilarityConfig(multi_label=True, threshold=0.5),
            )
        )
        assert [p.label for p in result.predictions] == ["cat"]

    async def test_multi_label_no_match_returns_unknown(self):
        scorer = SimilarityScorer()
        chunk_vectors = [(_chunk(), [1.0, 0.0])]
        label_vectors = {"dog": [0.0, 1.0]}
        result = await scorer.arun(
            SimilarityInput(
                chunk_vectors=chunk_vectors,
                label_vectors=label_vectors,
                config=SimilarityConfig(multi_label=True, threshold=0.99),
            )
        )
        assert result.predictions[0].label == "unknown"

    async def test_top_k_averages_best_scores_across_multiple_chunks(self):
        scorer = SimilarityScorer()
        chunk_vectors = [
            (_chunk("a"), [1.0, 0.0]),
            (_chunk("b"), [0.9, 0.1]),
        ]
        label_vectors = {"cat": [1.0, 0.0]}
        result = await scorer.arun(
            SimilarityInput(
                chunk_vectors=chunk_vectors,
                label_vectors=label_vectors,
                config=SimilarityConfig(top_k=1),
            )
        )
        assert result.top.label == "cat"
        assert result.top.score.value == 1.0

    async def test_euclidean_metric_bounds(self):
        scorer = SimilarityScorer()
        chunk_vectors = [(_chunk(), [1.0, 0.0])]
        label_vectors = {"cat": [1.0, 0.0]}
        result = await scorer.arun(
            SimilarityInput(
                chunk_vectors=chunk_vectors,
                label_vectors=label_vectors,
                config=SimilarityConfig(metric=SimilarityMetric.EUCLIDEAN),
            )
        )
        assert result.top.score.low == 0.0
        assert result.top.score.high == 1.0


class TestSimilarityConfig:
    def test_defaults(self):
        config = SimilarityConfig()
        assert config.top_k == 5
        assert config.threshold is None
        assert config.multi_label is False
        assert config.unknown_label == "unknown"
        assert config.metric == SimilarityMetric.COSINE

    def test_accepts_positional_arguments(self):
        config = SimilarityConfig(2, 0.5, True, "none", SimilarityMetric.DOT)
        assert (config.top_k, config.threshold, config.multi_label) == (2, 0.5, True)
        assert config.unknown_label == "none"
        assert config.metric == SimilarityMetric.DOT

    @pytest.mark.parametrize("top_k", [0, -1])
    def test_rejects_top_k_below_one(self, top_k):
        with pytest.raises(ValidationError, match="top_k"):
            SimilarityConfig(top_k=top_k)

    def test_rejects_unknown_metric(self):
        with pytest.raises(ValidationError, match="metric"):
            SimilarityConfig(metric="hamming")

    def test_is_frozen(self):
        config = SimilarityConfig()
        with pytest.raises(FrozenInstanceError):
            config.top_k = 3
