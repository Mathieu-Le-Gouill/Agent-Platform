from unittest.mock import AsyncMock, MagicMock, patch

from agent_platform.integrations.classification.providers.openai_classifier import (
    OpenAIZeroShotClassifier,
)
from agent_platform.models.chunk import TextChunk


@patch(
    "agent_platform.integrations.classification.providers.openai_classifier.AsyncOpenAI"
)
async def test_classify_empty_label_skips(mock_async_openai):
    classifier = OpenAIZeroShotClassifier(api_key="test-key")

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = ""

    mock_create = AsyncMock(return_value=mock_response)
    mock_async_openai.return_value.chat.completions.create = mock_create

    items = [TextChunk(text="Hello", metadata={})]
    result = await classifier.classify(items, candidate_labels=["pos", "neg"])

    assert result == []
