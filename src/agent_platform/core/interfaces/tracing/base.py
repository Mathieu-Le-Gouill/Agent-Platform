from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import AbstractContextManager

from agent_platform.core.interfaces.tracing.span import AttributeValue, Span

__all__ = ["BaseTracer"]


class BaseTracer(ABC):
    """Creates spans. One implementation per tracing backend.

    The abstraction sits at the tracer level, not at the span-context level:
    an implementation is expected to delegate parent/child nesting and
    cross-task context propagation to its backend rather than reimplement
    them, so `start_span` is the only thing a backend has to supply.
    """

    @abstractmethod
    def start_span(
        self, name: str, attributes: Mapping[str, AttributeValue] | None = None
    ) -> AbstractContextManager[Span]:
        """Start a span named `name`, ending it when the context manager exits.

        The returned span becomes the parent of any span started inside the
        `with` block, including from other coroutines it awaits.
        """
