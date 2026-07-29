import math

import pytest

from agent_platform.core.schemas.enums import SimilarityMetric
from agent_platform.core.similarity import compute_similarity, similarity_bounds


class TestComputeSimilarity:
    def test_cosine_identical_vectors(self):
        assert (
            compute_similarity([1.0, 0.0], [1.0, 0.0], SimilarityMetric.COSINE) == 1.0
        )

    def test_cosine_orthogonal_vectors(self):
        assert (
            compute_similarity([1.0, 0.0], [0.0, 1.0], SimilarityMetric.COSINE) == 0.0
        )

    def test_cosine_zero_vector_returns_zero(self):
        assert (
            compute_similarity([0.0, 0.0], [1.0, 1.0], SimilarityMetric.COSINE) == 0.0
        )

    def test_dot_product(self):
        assert compute_similarity([1.0, 2.0], [3.0, 4.0], SimilarityMetric.DOT) == 11.0

    def test_euclidean_identical_vectors(self):
        assert (
            compute_similarity([1.0, 1.0], [1.0, 1.0], SimilarityMetric.EUCLIDEAN)
            == 1.0
        )

    def test_euclidean_bounded_between_zero_and_one(self):
        result = compute_similarity([0.0, 0.0], [3.0, 4.0], SimilarityMetric.EUCLIDEAN)
        assert 0.0 < result < 1.0
        assert result == pytest.approx(1.0 / (1.0 + 5.0))

    def test_manhattan_identical_vectors(self):
        assert (
            compute_similarity([1.0, 1.0], [1.0, 1.0], SimilarityMetric.MANHATTAN)
            == 1.0
        )

    def test_manhattan_bounded_between_zero_and_one(self):
        result = compute_similarity([0.0], [3.0], SimilarityMetric.MANHATTAN)
        assert result == pytest.approx(1.0 / (1.0 + 3.0))

    def test_default_metric_is_cosine(self):
        assert compute_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0

    def test_cosine_matches_manual_computation(self):
        a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
        expected = (1 * 4 + 2 * 5 + 3 * 6) / (
            math.sqrt(1 + 4 + 9) * math.sqrt(16 + 25 + 36)
        )
        assert compute_similarity(a, b, SimilarityMetric.COSINE) == pytest.approx(
            expected
        )


class TestSimilarityBounds:
    def test_cosine_bounds(self):
        assert similarity_bounds(SimilarityMetric.COSINE) == (-1.0, 1.0)

    def test_dot_bounds(self):
        assert similarity_bounds(SimilarityMetric.DOT) == (float("-inf"), float("inf"))

    def test_euclidean_bounds(self):
        assert similarity_bounds(SimilarityMetric.EUCLIDEAN) == (0.0, 1.0)

    def test_manhattan_bounds(self):
        assert similarity_bounds(SimilarityMetric.MANHATTAN) == (0.0, 1.0)
