from __future__ import annotations

import threading

from agent_platform.core.interfaces.tracing.base import BaseTracer
from agent_platform.core.interfaces.tracing.noop import NoOpTracer
from agent_platform.core.interfaces.tracing.span import Span

__all__ = [
    "current_span",
    "get_tracer",
    "reset_tracer",
    "set_tracer",
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


def current_span() -> Span:
    """The span currently in scope, or an inert one when tracing is off.

    Exposed for correlating a trace with data submitted to a backend out of
    band (attaching an eval score or user feedback to a Langfuse/LangSmith
    trace after the fact) through `trace_id`/`span_id`. To describe what
    happened, use an `Observation` instead of writing attributes here.
    """
    return get_tracer().current_span()
