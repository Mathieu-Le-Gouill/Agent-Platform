from unittest.mock import AsyncMock

import pytest

from agent_platform.components.llm_classifier.component import LLMClassifier
from agent_platform.components.llm_classifier.config import LLMClassifierConfig
from agent_platform.core.interfaces.llm.response import LLMResponse
from agent_platform.core.schemas.document import TextDocument
from agent_platform.core.schemas.message import AssistantMessage, ToolCall
from agent_platform.core.schemas.token import TokenUsage


def _llm_response(tool_calls: list[ToolCall] | None = None) -> LLMResponse:
    return LLMResponse(
        message=AssistantMessage(content="", tool_calls=tool_calls or []),
        usage=TokenUsage(),
        model="test-model",
    )


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    return llm


class TestLLMClassifier:
    async def test_single_label_extracts_prediction(self, mock_llm):
        mock_llm.agenerate = AsyncMock(
            return_value=_llm_response(
                [
                    ToolCall(
                        id="1",
                        name="record_classification",
                        arguments={"label": "cat", "score": {"value": 0.9}},
                    )
                ]
            )
        )
        classifier = LLMClassifier(llm=mock_llm)
        docs = [TextDocument(text="meow")]
        response = await classifier.arun((docs, ["cat", "dog"], LLMClassifierConfig()))

        assert len(response.results) == 1
        assert response.results[0].label == "cat"
        assert response.results[0].score.value == 0.9

    async def test_multi_label_extracts_predictions(self, mock_llm):
        mock_llm.agenerate = AsyncMock(
            return_value=_llm_response(
                [
                    ToolCall(
                        id="1",
                        name="record_classification",
                        arguments={
                            "labels": ["cat", "dog"],
                            "scores": [{"value": 0.9}, {"value": 0.1}],
                        },
                    )
                ]
            )
        )
        classifier = LLMClassifier(llm=mock_llm)
        docs = [TextDocument(text="meow")]
        config = LLMClassifierConfig(multi_label=True)
        response = await classifier.arun((docs, ["cat", "dog"], config))

        assert [p.label for p in response.results[0].predictions] == ["cat", "dog"]

    async def test_no_matching_tool_call_returns_empty_predictions(self, mock_llm):
        mock_llm.agenerate = AsyncMock(
            return_value=_llm_response(
                [ToolCall(id="1", name="other_tool", arguments={})]
            )
        )
        classifier = LLMClassifier(llm=mock_llm)
        docs = [TextDocument(text="meow")]
        response = await classifier.arun((docs, ["cat"], LLMClassifierConfig()))

        assert response.results[0].predictions == []

    async def test_no_message_returns_empty_predictions(self, mock_llm):
        mock_llm.agenerate = AsyncMock(
            return_value=LLMResponse(message=None, usage=TokenUsage(), model="m")
        )
        classifier = LLMClassifier(llm=mock_llm)
        docs = [TextDocument(text="meow")]
        response = await classifier.arun((docs, ["cat"], LLMClassifierConfig()))

        assert response.results[0].predictions == []

    async def test_invalid_score_shape_is_logged_and_skipped(self, mock_llm):
        mock_llm.agenerate = AsyncMock(
            return_value=_llm_response(
                [
                    ToolCall(
                        id="1",
                        name="record_classification",
                        arguments={"label": "cat", "score": "not-a-score"},
                    )
                ]
            )
        )
        classifier = LLMClassifier(llm=mock_llm)
        docs = [TextDocument(text="meow")]
        response = await classifier.arun((docs, ["cat"], LLMClassifierConfig()))

        assert response.results[0].predictions == []

    async def test_missing_score_argument_raises_key_error(self, mock_llm):
        mock_llm.agenerate = AsyncMock(
            return_value=_llm_response(
                [
                    ToolCall(
                        id="1",
                        name="record_classification",
                        arguments={"label": "cat"},
                    )
                ]
            )
        )
        classifier = LLMClassifier(llm=mock_llm)
        docs = [TextDocument(text="meow")]
        with pytest.raises(KeyError):
            await classifier.arun((docs, ["cat"], LLMClassifierConfig()))

    async def test_multiple_documents_produce_multiple_results(self, mock_llm):
        mock_llm.agenerate = AsyncMock(
            return_value=_llm_response(
                [
                    ToolCall(
                        id="1",
                        name="record_classification",
                        arguments={"label": "cat", "score": {"value": 0.9}},
                    )
                ]
            )
        )
        classifier = LLMClassifier(llm=mock_llm)
        docs = [TextDocument(text="meow"), TextDocument(text="woof")]
        response = await classifier.arun((docs, ["cat", "dog"], LLMClassifierConfig()))

        assert len(response.results) == 2
        assert mock_llm.agenerate.await_count == 2

    async def test_uses_default_config_when_input_config_falsy(self, mock_llm):
        mock_llm.agenerate = AsyncMock(return_value=_llm_response())
        classifier = LLMClassifier(llm=mock_llm, config=LLMClassifierConfig())
        docs = [TextDocument(text="meow")]
        response = await classifier.arun((docs, ["cat"], None))

        assert len(response.results) == 1
