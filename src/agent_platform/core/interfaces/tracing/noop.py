from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager

from agent_platform.core.interfaces.tracing.base import BaseTracer
from agent_platform.core.interfaces.tracing.span import AttributeValue, Span

__all__ = ["NoOpSpan", "NoOpTracer"]


class NoOpSpan:
    """A `Span` that discards everything it is given."""

    def set_attribute(self, key: str, value: AttributeValue) -> None:
        pass

    def record_error(self, exc: Exception, *, record_exception: bool = True) -> None:
        pass

    @property
    def trace_id(self) -> str | None:
        return None

    @property
    def span_id(self) -> str | None:
        return None


class NoOpTracer(BaseTracer):
    """The default tracer: records nothing, costs nothing.

    Keeping a working no-op in `core` is what lets every call site use
    `observe` unconditionally, with no "is tracing enabled" branch and no
    tracing dependency installed.
    """

    _SPAN = NoOpSpan()

    @contextmanager
    def start_span(
        self, name: str, attributes: Mapping[str, AttributeValue] | None = None
    ) -> Iterator[Span]:
        yield self._SPAN

    def current_span(self) -> Span:
        return self._SPAN
