from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from agent_platform.models.chunk import AudioChunk, TextChunk
from agent_platform.models.document import TextDocument
from agent_platform.models.enums import DataType, DocumentFormat
from agent_platform.models.message import (
    SystemMessage,
    UserMessage,
    Prompt,
)
from agent_platform.models.token import TokenUsage


@pytest.fixture
def fake_text_chunk() -> TextChunk:
    return TextChunk(
        id=uuid4(),
        text="Hello world",
        index=0,
        metadata={"source": "test", "language": "en"},
    )


@pytest.fixture
def fake_text_document() -> TextDocument:
    return TextDocument(
        id=uuid4(),
        text="Hello world\nThis is a test document.",
        source="test.txt",
        format=DocumentFormat.TXT,
    )


@pytest.fixture
def fake_audio_chunk() -> AudioChunk:
    return AudioChunk(
        id=uuid4(),
        data=b"\x00\x01\x02\x03",
        sample_rate=16000,
        start=0,
        end=16000,
        channels=1,
        dtype=DataType.FLOAT32,
    )


@pytest.fixture
def fake_prompt() -> Prompt:
    return Prompt(
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="Hello!"),
        ]
    )


@pytest.fixture
def fake_token_usage() -> TokenUsage:
    return TokenUsage(input_tokens=10, output_tokens=20)


@pytest.fixture
def mock_langchain_response() -> MagicMock:
    response = MagicMock()
    response.content = "Hello, I am an AI assistant."
    response.tool_calls = []
    response.usage_metadata = {"input_tokens": 10, "output_tokens": 20}
    return response
