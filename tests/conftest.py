from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from agent_platform.core.schemas.chunk import AudioChunk, TextChunk
from agent_platform.core.schemas.document import TextDocument
from agent_platform.core.schemas.enums import DataType, DocumentFormat
from agent_platform.core.schemas.message import (
    Prompt,
    SystemMessage,
    UserMessage,
)
from agent_platform.core.schemas.token import TokenUsage


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


# ---------------------------------------------------------------------------
# Agent test helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.agenerate = AsyncMock()
    return llm


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if not any(item.iter_markers(name="integration")):
            item.add_marker("unit")
