from agent_platform.core.interfaces.classification.response import (
    ClassificationPrediction,
    ClassificationResponse,
    ClassificationResult,
)
from agent_platform.core.schemas.score import Score


def _prediction(label: str, value: float) -> ClassificationPrediction:
    return ClassificationPrediction(label=label, score=Score(value=value))


class TestClassificationResult:
    def test_top_returns_first_prediction(self):
        result = ClassificationResult(
            predictions=[_prediction("cat", 0.9), _prediction("dog", 0.1)]
        )
        assert result.top.label == "cat"

    def test_top_returns_none_when_empty(self):
        result = ClassificationResult(predictions=[])
        assert result.top is None

    def test_label_delegates_to_top(self):
        result = ClassificationResult(predictions=[_prediction("cat", 0.9)])
        assert result.label == "cat"

    def test_label_none_when_empty(self):
        result = ClassificationResult(predictions=[])
        assert result.label is None

    def test_score_delegates_to_top(self):
        result = ClassificationResult(predictions=[_prediction("cat", 0.9)])
        assert result.score.value == 0.9

    def test_score_none_when_empty(self):
        result = ClassificationResult(predictions=[])
        assert result.score is None


class TestClassificationResponse:
    def test_holds_multiple_results(self):
        response = ClassificationResponse(
            results=[
                ClassificationResult(predictions=[_prediction("cat", 0.9)]),
                ClassificationResult(predictions=[_prediction("spam", 0.6)]),
            ]
        )
        assert len(response.results) == 2
        assert response.results[0].label == "cat"
