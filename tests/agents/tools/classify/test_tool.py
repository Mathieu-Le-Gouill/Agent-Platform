from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from agent_platform.agents.tools import ClassifyInput, ClassifyTool, ToolError
from agent_platform.core.interfaces.classification.response import (
    ClassificationPrediction,
    ClassificationResult,
)
from agent_platform.core.schemas.score import Score


@pytest.fixture
def mock_classifier():
    classifier = AsyncMock()
    classifier.classify = AsyncMock(
        return_value=ClassificationResult(
            predictions=[
                ClassificationPrediction(label="sports", score=Score(value=0.8)),
                ClassificationPrediction(label="politics", score=Score(value=0.2)),
            ]
        )
    )
    return classifier


@pytest.fixture
def tool(mock_classifier):
    return ClassifyTool(classifier=mock_classifier)


class TestClassifyInput:
    def test_valid_input(self):
        inp = ClassifyInput(
            text="Real Madrid won", candidate_labels=["sports", "politics"]
        )
        assert inp.text == "Real Madrid won"
        assert inp.candidate_labels == ["sports", "politics"]

    def test_empty_text_raises(self):
        with pytest.raises(ValidationError):
            ClassifyInput(text="", candidate_labels=["sports"])

    def test_empty_candidate_labels_raises(self):
        with pytest.raises(ValidationError):
            ClassifyInput(text="hi", candidate_labels=[])


class TestClassifyTool:
    def test_name_and_description(self, tool):
        assert tool.name == "classify"
        assert tool.description

    async def test_run_success(self, tool, mock_classifier):
        result = await tool.run(
            text="Real Madrid won", candidate_labels=["sports", "politics"]
        )
        assert result.label == "sports"
        mock_classifier.classify.assert_awaited_once_with(
            "Real Madrid won", ["sports", "politics"]
        )

    async def test_run_provider_error_wrapped(self, tool, mock_classifier):
        mock_classifier.classify = AsyncMock(side_effect=RuntimeError("boom"))
        with pytest.raises(ToolError, match="Classification failed"):
            await tool.run(text="hi", candidate_labels=["a", "b"])

    async def test_run_missing_labels_raises(self, tool):
        with pytest.raises(ValidationError):
            await tool.run(text="hi")
