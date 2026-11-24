from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

__all__ = ["ToolCallContext", "current_tool_call"]


@dataclass(frozen=True)
class ToolCallContext:
    """What a tool can read about the call it is running.

    `idempotency_key` identifies one tool call. Inside an agent it is the id the
    model gave the call, so the corrective retry of a turn keeps it; under a
    workflow `tool_node` it is built from the run, node and step, so a resumed
    run keeps it. A tool with a side effect deduplicates on it.
    """

    tool_name: str
    idempotency_key: str


_current: ContextVar[ToolCallContext | None] = ContextVar(
    "current_tool_call", default=None
)


def current_tool_call() -> ToolCallContext | None:
    """Return the call the running tool was started for, or `None`.

    Only set while a `ToolRegistry` runs the tool, in the task that runs it.
    """
    return _current.get()


@contextmanager
def bind_tool_call(context: ToolCallContext) -> Iterator[None]:
    """Make `context` the current tool call for the duration of the block."""
    token = _current.set(context)
    try:
        yield
    finally:
        _current.reset(token)
