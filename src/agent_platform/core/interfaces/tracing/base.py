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

    @abstractmethod
    def current_span(self) -> Span:
        """The innermost span currently in scope, or an inert span if none is.

        Lets code nested under a span annotate it without that span being
        threaded through every signature in between (a workflow node
        stamping the node span, an agent stamping its loop span). Never
        returns `None`, so call sites annotate unconditionally.
        """
