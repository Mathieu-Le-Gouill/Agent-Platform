from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from agent_platform.core import observability
from agent_platform.core.interfaces.tracing.base import BaseTracer
from agent_platform.core.interfaces.tracing.noop import NoOpSpan
from agent_platform.core.interfaces.tracing.span import AttributeValue, Span
from agent_platform.core.observability import ERROR_TYPE_KEY
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
    llm.stream = MagicMock()
    return llm


@pytest.fixture
def no_retry_sleep(monkeypatch) -> None:
    """Skip the real sleep delay in core.retry's retry-with-backoff helpers (async and sync)."""
    import agent_platform.core.retry as retry_mod

    monkeypatch.setattr(retry_mod.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr(retry_mod.time, "sleep", MagicMock())


@dataclass
class RecordedSpan:
    """One finished span, as the platform described it (platform key names)."""

    name: str
    attributes: dict[str, AttributeValue] = field(default_factory=dict)
    error: Exception | None = None
    record_exception: bool | None = None
    parent: RecordedSpan | None = None


class _RecordingSpan:
    def __init__(self, record: RecordedSpan) -> None:
        self._record = record

    def set_attribute(self, key: str, value: AttributeValue) -> None:
        self._record.attributes[key] = value

    def record_error(self, exc: Exception, *, record_exception: bool = True) -> None:
        self._record.error = exc
        self._record.record_exception = record_exception
        self._record.attributes[ERROR_TYPE_KEY] = type(exc).__qualname__

    @property
    def trace_id(self) -> str | None:
        return None

    @property
    def span_id(self) -> str | None:
        return None


class RecordingTracer(BaseTracer):
    """Keeps each span in finish order, with no tracing backend involved."""

    def __init__(self) -> None:
        self._finished: list[RecordedSpan] = []
        self._open: ContextVar[_RecordingSpan | None] = ContextVar(
            "recording_tracer_open", default=None
        )

    def get_finished_spans(self) -> tuple[RecordedSpan, ...]:
        return tuple(self._finished)

    @contextmanager
    def start_span(
        self, name: str, attributes: Mapping[str, AttributeValue] | None = None
    ) -> Iterator[Span]:
        parent = self._open.get()
        record = RecordedSpan(
            name,
            dict(attributes or {}),
            parent=parent._record if parent is not None else None,
        )
        span = _RecordingSpan(record)
        token = self._open.set(span)
        try:
            yield span
        finally:
            self._open.reset(token)
            self._finished.append(record)

    def current_span(self) -> Span:
        return self._open.get() or NoOpSpan()


@pytest.fixture
def recorded_spans():
    """Install a backend-free recorder as `observe()`'s tracer so a test can
    assert on the spans a call produced, by platform key. Root-level (not
    `tests/core/` only) since span attributes - `conversation.id` in
    particular - are asserted on from `agents/`, not just `core/`."""
    tracer = RecordingTracer()
    observability.set_tracer(tracer)
    yield tracer
    observability.reset_tracer()


@pytest.fixture(autouse=True)
def clear_langfuse_env(monkeypatch):
    """Keep a developer's shell keys from reaching the suite."""
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if not any(item.iter_markers(name="integration")):
            item.add_marker("unit")
