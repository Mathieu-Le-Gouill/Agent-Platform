import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_platform.integrations.classification.providers.zero_shot import (
    ZeroShotClassifier,
)
from agent_platform.integrations.classification.providers.openai_classifier import (
    OpenAIZeroShotClassifier,
)
from agent_platform.models.chunk import TextChunk


# ---------------------------------------------------------------------------
# ZeroShotClassifier._infer_labels
# ---------------------------------------------------------------------------


def test_infer_labels_from_metadata():
    items = [
        TextChunk(text="a", metadata={"label": "pos"}),
        TextChunk(text="b", metadata={"label": "neg"}),
        TextChunk(text="c", metadata={"label": "pos"}),
    ]
    labels = ZeroShotClassifier._infer_labels(items)
    assert labels == ["pos", "neg"]


def test_infer_labels_no_metadata_falls_back():
    items = [
        TextChunk(text="a", metadata={}),
        TextChunk(text="b", metadata={}),
    ]
    labels = ZeroShotClassifier._infer_labels(items)
    assert labels == ["positive", "negative", "neutral"]


def test_infer_labels_mixed():
    items = [
        TextChunk(text="a", metadata={"label": "spam"}),
        TextChunk(text="b", metadata={}),
    ]
    labels = ZeroShotClassifier._infer_labels(items)
    assert labels == ["spam"]


def test_infer_labels_preserves_order():
    items = [
        TextChunk(text="a", metadata={"label": "c"}),
        TextChunk(text="b", metadata={"label": "a"}),
        TextChunk(text="c", metadata={"label": "b"}),
    ]
    labels = ZeroShotClassifier._infer_labels(items)
    assert labels == ["c", "a", "b"]


# ---------------------------------------------------------------------------
# ZeroShotClassifier constructor defaults
# ---------------------------------------------------------------------------


def test_zero_shot_constructor_defaults():
    classifier = ZeroShotClassifier()
    assert classifier._model_name == "facebook/bart-large-mnli"
    assert classifier._device == -1
    assert classifier._hypothesis_template == "This example is {}"
    assert classifier._pipeline is None


# ---------------------------------------------------------------------------
# OpenAIZeroShotClassifier constructor defaults
# ---------------------------------------------------------------------------


@patch(
    "agent_platform.integrations.classification.providers.openai_classifier.AsyncOpenAI"
)
def test_openai_zero_shot_constructor_defaults(mock_async_openai):
    classifier = OpenAIZeroShotClassifier(api_key="test-key")
    mock_async_openai.assert_called_once_with(api_key="test-key")
    assert classifier._model == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Empty text handling — ZeroShotClassifier
# ---------------------------------------------------------------------------


async def test_zero_shot_empty_text():
    classifier = ZeroShotClassifier()
    classifier._pipeline = MagicMock()
    result = await classifier.classify(
        [TextChunk(text="", metadata={}), TextChunk(text="   ", metadata={})],
        candidate_labels=["a", "b"],
    )
    assert result == ["", ""]
    classifier._pipeline.assert_not_called()


# ---------------------------------------------------------------------------
# Empty text handling — OpenAIZeroShotClassifier
# ---------------------------------------------------------------------------


@patch(
    "agent_platform.integrations.classification.providers.openai_classifier.AsyncOpenAI"
)
async def test_openai_zero_shot_empty_text(mock_async_openai):
    classifier = OpenAIZeroShotClassifier(api_key="test-key")
    result = await classifier.classify(
        [TextChunk(text="", metadata={}), TextChunk(text="   ", metadata={})],
    )
    assert result == ["", ""]
    mock_async_openai.return_value.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# ZeroShotClassifier.classify flow with mocked pipeline
# ---------------------------------------------------------------------------


async def test_zero_shot_classify_flow():
    classifier = ZeroShotClassifier()
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = {"labels": ["pos", "neg"], "scores": [0.95, 0.05]}
    classifier._pipeline = mock_pipeline

    async def _mock_run_in_executor(executor, fn):
        return fn()

    loop = asyncio.get_event_loop()
    with patch.object(loop, "run_in_executor", side_effect=_mock_run_in_executor):
        items = [TextChunk(text="Great product!", metadata={})]
        result = await classifier.classify(items, candidate_labels=["pos", "neg"])

    assert result == ["pos"]
    mock_pipeline.assert_called_once_with(
        "Great product!",
        candidate_labels=["pos", "neg"],
        hypothesis_template="This example is {}",
    )


async def test_zero_shot_classify_empty_items():
    classifier = ZeroShotClassifier()
    classifier._pipeline = MagicMock()
    result = await classifier.classify([], candidate_labels=["a", "b"])
    assert result == []
    classifier._pipeline.assert_not_called()


# ---------------------------------------------------------------------------
# OpenAIZeroShotClassifier.classify flow with mocked client
# ---------------------------------------------------------------------------


@patch(
    "agent_platform.integrations.classification.providers.openai_classifier.AsyncOpenAI"
)
async def test_openai_zero_shot_classify_flow(mock_async_openai):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "positive"

    mock_create = AsyncMock(return_value=mock_response)
    mock_async_openai.return_value.chat.completions.create = mock_create

    classifier = OpenAIZeroShotClassifier(api_key="test-key")
    items = [TextChunk(text="Great!", metadata={})]
    result = await classifier.classify(items, candidate_labels=["positive", "negative"])

    assert result == ["positive"]
    mock_create.assert_called_once_with(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a text classifier. Classify the following text into"
                    " exactly one of these categories: positive, negative."
                    " Respond with ONLY the category name, nothing else."
                ),
            },
            {"role": "user", "content": "Great!"},
        ],
        temperature=0.0,
        max_tokens=50,
    )
