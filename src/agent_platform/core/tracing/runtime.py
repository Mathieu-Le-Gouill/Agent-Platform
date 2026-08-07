from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

from agent_platform.core.interfaces.tracing.base import BaseTracer
from agent_platform.core.interfaces.tracing.noop import NoOpTracer
from agent_platform.core.interfaces.tracing.span import AttributeValue, Span

__all__ = [
    "get_tracer",
    "reset_tracer",
    "set_tracer",
    "traced_span",
]

_tracer_lock = threading.Lock()
_tracer: BaseTracer = NoOpTracer()


def set_tracer(tracer: BaseTracer) -> None:
    """Install `tracer` as the process-wide tracer.

    Call once at startup, before any span is created. Applications wire this
    from `config.container.build_tracer()`; nothing in `core` selects a
    backend, which is what keeps this layer free of any tracing dependency.
    """
    global _tracer
    with _tracer_lock:
        _tracer = tracer


def reset_tracer() -> None:
    """Restore the no-op default. Mainly for tests and for shutting tracing off."""
    set_tracer(NoOpTracer())


def get_tracer() -> BaseTracer:
    return _tracer


@contextmanager
def traced_span(
    name: str, /, attributes: Mapping[str, AttributeValue | None] | None = None
) -> Iterator[Span]:
    """Start a span named `name` with `attributes`, recording latency and errors.

    Safe to use unconditionally: with no tracer installed this is a
    pass-through over `NoOpTracer`. Attributes whose value is `None` are
    dropped rather than exported, so call sites can pass optional values
    (`conversation_id` and friends) without branching.
    """
    resolved: dict[str, AttributeValue] = {
        key: value for key, value in (attributes or {}).items() if value is not None
    }
    with get_tracer().start_span(name, resolved) as span:
        try:
            yield span
        except Exception as exc:
            span.record_error(exc)
            raise
