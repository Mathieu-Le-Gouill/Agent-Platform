from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TypeVar

from agent_platform.core.interfaces.tracing.span import AttributeValue, Span
from agent_platform.core.observability.facts import Observation, flatten
from agent_platform.core.observability.tracer import current_span, get_tracer

__all__ = ["current", "emit", "observe"]

ObservationT = TypeVar("ObservationT", bound=Observation)

# The observations whose spans are currently open, innermost last. Kept here
# rather than read back off the tracer because a backend's span carries
# attributes, not the typed object a call site wants to keep annotating.
_open: ContextVar[tuple[Observation, ...]] = ContextVar("open_observations")


@contextmanager
def observe(observation: ObservationT) -> Iterator[ObservationT]:
    """Run a unit of work as `observation`: one span, described by its facts.

    Safe to use unconditionally: with no tracer installed this is a
    pass-through over the no-op default. The observation is yielded back so
    the call site can keep filling it in as the work reveals more; whatever
    changed by the time the block exits is flushed onto the span before it
    ends. Duration is the span's own, so nothing here times the block.
    """
    start = flatten(observation)
    raised = False
    token = _open.set((*_open.get(()), observation))
    try:
        with get_tracer().start_span(observation.operation_name, start) as span:
            try:
                yield observation
            except Exception as exc:
                raised = True
                span.record_error(exc)
                raise
            finally:
                _flush(span, start, flatten(observation))
                # An error that propagated is already recorded, with its
                # stack trace; `observation.error` is only for one the call
                # site swallowed.
                if not raised and observation.error is not None:
                    span.record_error(observation.error, record_exception=False)
    finally:
        _open.reset(token)


def emit(observation: Observation) -> None:
    """Record `observation`'s facts on whatever span is already open.

    For facts that describe a step inside someone else's unit of work rather
    than a unit of work of their own (what a retrieval found, who a workflow
    handed off to, what a finished model call cost). There is no duration,
    because there is nothing to time.
    """
    span = current_span()
    for key, value in flatten(observation).items():
        span.set_attribute(key, value)


def current(kind: type[ObservationT]) -> ObservationT | None:
    """The innermost open observation of type `kind`, or `None`.

    Lets code nested under someone else's observation annotate it without
    that object being threaded through every signature in between.
    """
    for observation in reversed(_open.get(())):
        if isinstance(observation, kind):
            return observation
    return None


def _flush(
    span: Span,
    start: Mapping[str, AttributeValue],
    final: Mapping[str, AttributeValue],
) -> None:
    for key, value in final.items():
        if start.get(key) != value:
            span.set_attribute(key, value)
