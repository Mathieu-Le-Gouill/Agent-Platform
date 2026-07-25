from uuid import uuid4

from agent_platform.components.similarity_scorer import (
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
        scorer = SimilarityScorer(config=SimilarityConfig(threshold=0.9))
        chunk_vectors = [(_chunk(), [1.0, 0.0])]
        label_vectors = {"dog": [0.0, 1.0]}
        result = await scorer.arun(
            SimilarityInput(chunk_vectors=chunk_vectors, label_vectors=label_vectors)
        )
        assert result.predictions[0].label == "unknown"

    async def test_multi_label_returns_all_above_threshold(self):
        scorer = SimilarityScorer(
            config=SimilarityConfig(multi_label=True, threshold=0.5)
        )
        chunk_vectors = [(_chunk(), [1.0, 0.0])]
        label_vectors = {"cat": [1.0, 0.0], "dog": [0.0, 1.0]}
        result = await scorer.arun(
            SimilarityInput(chunk_vectors=chunk_vectors, label_vectors=label_vectors)
        )
        assert [p.label for p in result.predictions] == ["cat"]

    async def test_multi_label_no_match_returns_unknown(self):
        scorer = SimilarityScorer(
            config=SimilarityConfig(multi_label=True, threshold=0.99)
        )
        chunk_vectors = [(_chunk(), [1.0, 0.0])]
        label_vectors = {"dog": [0.0, 1.0]}
        result = await scorer.arun(
            SimilarityInput(chunk_vectors=chunk_vectors, label_vectors=label_vectors)
        )
        assert result.predictions[0].label == "unknown"

    async def test_top_k_averages_best_scores_across_multiple_chunks(self):
        scorer = SimilarityScorer(config=SimilarityConfig(top_k=1))
        chunk_vectors = [
            (_chunk("a"), [1.0, 0.0]),
            (_chunk("b"), [0.9, 0.1]),
        ]
        label_vectors = {"cat": [1.0, 0.0]}
        result = await scorer.arun(
            SimilarityInput(chunk_vectors=chunk_vectors, label_vectors=label_vectors)
        )
        assert result.top.label == "cat"
        assert result.top.score.value == 1.0

    async def test_euclidean_metric_bounds(self):
        scorer = SimilarityScorer(
            config=SimilarityConfig(metric=SimilarityMetric.EUCLIDEAN)
        )
        chunk_vectors = [(_chunk(), [1.0, 0.0])]
        label_vectors = {"cat": [1.0, 0.0]}
        result = await scorer.arun(
            SimilarityInput(chunk_vectors=chunk_vectors, label_vectors=label_vectors)
        )
        assert result.top.score.low == 0.0
        assert result.top.score.high == 1.0
