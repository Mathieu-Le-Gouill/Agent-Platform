from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeAlias, runtime_checkable

__all__ = ["AttributeValue", "Span"]

# The value types a span attribute may hold. Deliberately narrow: every
# tracing backend worth supporting (OTLP-based or not) ultimately serializes
# attributes to JSON-ish primitives, so accepting `Any` here would only defer
# a silent drop of unsupported values to export time. Structured payloads
# (prompts, tool arguments) are passed as pre-serialized JSON strings.
AttributeValue: TypeAlias = (
    str
    | bool
    | int
    | float
    | Sequence[str]
    | Sequence[bool]
    | Sequence[int]
    | Sequence[float]
)


@runtime_checkable
class Span(Protocol):
    """One timed unit of work, as seen by platform code.

    A `Protocol` rather than an ABC because implementations are thin adapters
    over a backend's own span object, which they wrap rather than subclass.

    Intentionally minimal: attributes plus error state cover every call site
    and every backend-specific convention the platform needs to express
    (`gen_ai.*` semantic conventions, vendor keys such as Langfuse's
    `langfuse.observation.*`). Span timing is the backend's responsibility,
    it starts when the span is created and ends when its context manager
    exits.
    """

    def set_attribute(self, key: str, value: AttributeValue) -> None: ...

    def record_error(self, exc: Exception, *, record_exception: bool = True) -> None:
        """Mark this span as failed because of `exc`.

        `record_exception=False` is for call sites that catch an error without
        re-raising it (e.g. a tool call swallowed into a `ToolResult`): the
        span is still tagged as failed, but no stack trace is attached since
        the exception never leaves that call site.
        """
        ...

    @property
    def trace_id(self) -> str | None:
        """Hex trace id, or `None` when tracing is not active.

        Exposed so callers can correlate a trace with data submitted to a
        backend out of band, e.g. attaching an eval score or user feedback to
        a Langfuse/LangSmith trace after the fact.
        """
        ...

    @property
    def span_id(self) -> str | None:
        """Hex span id, or `None` when tracing is not active."""
        ...
