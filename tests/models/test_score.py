import pytest

from agent_platform.models.score import Score, ScoreKind


class TestScoreFactoryMethods:
    def test_similarity_default_bounds(self):
        s = Score.similarity(0.75)
        assert s.value == 0.75
        assert s.kind == ScoreKind.SIMILARITY
        assert s.low == 0.0
        assert s.high == 1.0

    def test_similarity_custom_bounds(self):
        s = Score.similarity(5.0, low=0.0, high=10.0)
        assert s.value == 5.0
        assert s.low == 0.0
        assert s.high == 10.0

    def test_relevance(self):
        s = Score.relevance(0.9)
        assert s.value == 0.9
        assert s.kind == ScoreKind.RELEVANCE

    def test_confidence(self):
        s = Score.confidence(0.95)
        assert s.value == 0.95
        assert s.kind == ScoreKind.CONFIDENCE

    def test_quality(self):
        s = Score.quality(0.8)
        assert s.value == 0.8
        assert s.kind == ScoreKind.QUALITY

    def test_logit_default_kind(self):
        s = Score.logit(2.5)
        assert s.value == 2.5
        assert s.kind == ScoreKind.CONFIDENCE
        assert s.low == float("-inf")
        assert s.high == float("inf")

    def test_logit_custom_kind(self):
        s = Score.logit(1.2, kind=ScoreKind.SIMILARITY)
        assert s.kind == ScoreKind.SIMILARITY


class TestScoreNormalized:
    def test_normalized_default_bounds(self):
        s = Score.similarity(0.5)
        assert s.normalized == 0.5

    def test_normalized_custom_bounds(self):
        s = Score.similarity(5.0, low=0.0, high=10.0)
        assert s.normalized == 0.5

    def test_normalized_at_low(self):
        s = Score.similarity(0.0)
        assert s.normalized == 0.0

    def test_normalized_at_high(self):
        s = Score.similarity(1.0)
        assert s.normalized == 1.0

    def test_normalized_negative_low(self):
        s = Score(value=0.0, kind=ScoreKind.SIMILARITY, low=-1.0, high=1.0)
        assert s.normalized == 0.5

    def test_normalized_logit(self):
        s = Score.logit(100.0)
        assert s.normalized == 100.0


class TestScoreToPercentage:
    def test_to_percentage(self):
        s = Score.similarity(0.75)
        assert s.to_percentage() == 75.0

    def test_to_percentage_rounding(self):
        s = Score.similarity(0.33333)
        assert s.to_percentage() == 33.33

    def test_to_percentage_zero(self):
        s = Score.similarity(0.0)
        assert s.to_percentage() == 0.0

    def test_to_percentage_full(self):
        s = Score.similarity(1.0)
        assert s.to_percentage() == 100.0


class TestScoreComparison:
    def test_exceeds_true(self):
        s = Score.similarity(0.9)
        assert s.exceeds(0.5) is True

    def test_exceeds_false(self):
        s = Score.similarity(0.3)
        assert s.exceeds(0.5) is False

    def test_exceeds_equal_not_exceeding(self):
        s = Score.similarity(0.5)
        assert s.exceeds(0.5) is False


class TestScoreHighConfidence:
    def test_is_high_confidence_default(self):
        s = Score.confidence(0.9)
        assert s.is_high_confidence() is True

    def test_is_high_confidence_below_threshold(self):
        s = Score.confidence(0.7)
        assert s.is_high_confidence() is False

    def test_is_high_confidence_custom_threshold(self):
        s = Score.confidence(0.95)
        assert s.is_high_confidence(threshold=0.9) is True

    def test_is_high_confidence_uses_normalized(self):
        s = Score(value=80.0, kind=ScoreKind.CONFIDENCE, low=0.0, high=100.0)
        assert s.is_high_confidence(threshold=0.75) is True

    def test_is_high_confidence_at_threshold(self):
        s = Score.confidence(0.85)
        assert s.is_high_confidence(threshold=0.85) is False


class TestScoreBoundsValidation:
    def test_low_greater_than_high_raises(self):
        with pytest.raises(ValueError, match="Score bounds invalid"):
            Score(value=0.5, kind=ScoreKind.SIMILARITY, low=1.0, high=0.0)

    def test_low_equal_high_raises(self):
        with pytest.raises(ValueError, match="Score bounds invalid"):
            Score(value=0.5, kind=ScoreKind.SIMILARITY, low=0.5, high=0.5)

    def test_value_below_low_raises(self):
        with pytest.raises(ValueError, match="outside"):
            Score(value=-0.1, kind=ScoreKind.SIMILARITY, low=0.0, high=1.0)

    def test_value_above_high_raises(self):
        with pytest.raises(ValueError, match="outside"):
            Score(value=1.1, kind=ScoreKind.SIMILARITY, low=0.0, high=1.0)

    def test_value_at_low_is_valid(self):
        s = Score(value=0.0, kind=ScoreKind.SIMILARITY, low=0.0, high=1.0)
        assert s.value == 0.0

    def test_value_at_high_is_valid(self):
        s = Score(value=1.0, kind=ScoreKind.SIMILARITY, low=0.0, high=1.0)
        assert s.value == 1.0

    def test_logit_unbounded(self):
        s = Score.logit(999.0)
        assert s.value == 999.0

    def test_logit_negative_inf(self):
        s = Score.logit(-999.0)
        assert s.value == -999.0
