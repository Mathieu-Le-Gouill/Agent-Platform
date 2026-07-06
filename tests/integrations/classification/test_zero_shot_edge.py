import asyncio
from unittest.mock import MagicMock, patch

import pytest

from agent_platform.integrations.classification.providers.zero_shot import (
    ZeroShotClassifier,
)
from agent_platform.models.chunk import TextChunk


async def test_load_early_return():
    classifier = ZeroShotClassifier()
    classifier._pipeline = MagicMock()
    original = classifier._pipeline
    await classifier._load()
    assert classifier._pipeline is original


async def test_classify_raises_when_pipeline_none():
    classifier = ZeroShotClassifier()
    classifier._pipeline = None
    with pytest.raises(
        RuntimeError, match="Failed to load Zero Shot Classification pipeline"
    ):
        await classifier.classify(
            [TextChunk(text="test", metadata={})],
            candidate_labels=["a", "b"],
        )


async def test_classify_without_candidate_labels():
    classifier = ZeroShotClassifier()
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = {"labels": ["pos", "neg"], "scores": [0.9, 0.1]}
    classifier._pipeline = mock_pipeline

    async def _mock_run_in_executor(executor, fn):
        return fn()

    loop = asyncio.get_event_loop()
    with patch.object(loop, "run_in_executor", side_effect=_mock_run_in_executor):
        items = [TextChunk(text="Nice!", metadata={"label": "pos"})]
        result = await classifier.classify(items)
    assert result == ["pos"]
    mock_pipeline.assert_called_once_with(
        "Nice!",
        candidate_labels=["pos"],
        hypothesis_template="This example is {}",
    )


async def test_classify_handles_empty_output():
    classifier = ZeroShotClassifier()
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = None
    classifier._pipeline = mock_pipeline

    async def _mock_run_in_executor(executor, fn):
        return fn()

    loop = asyncio.get_event_loop()
    with patch.object(loop, "run_in_executor", side_effect=_mock_run_in_executor):
        items = [TextChunk(text="Great!", metadata={})]
        result = await classifier.classify(items, candidate_labels=["a", "b"])
    assert result == []
