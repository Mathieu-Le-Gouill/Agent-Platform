from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType

__all__ = ["ToolCallContext", "current_tool_call"]


@dataclass(frozen=True)
class ToolCallContext:
    """What a tool can read about the call it is running.

    `idempotency_key` identifies one tool call. Inside an agent it is the id the
    model gave the call; when the corrective retry re-emits a call that failed,
    unchanged, the retried call keeps the failed call's key (a call that
    succeeded is not run again, and a changed or new call gets its own id).
    Under a workflow `tool_node` it is built from the run, node and step, so a
    resumed run keeps it; a resumed `agent_node` still gets new keys. A tool with a side effect deduplicates on it.
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


_key_overrides: ContextVar[Mapping[str, str]] = ContextVar(
    "idempotency_key_overrides", default=MappingProxyType({})
)


def idempotency_key_for(call_id: str) -> str:
    """Return the idempotency key for the call `call_id`: its own id unless a
    `bind_idempotency_keys` block maps it to an earlier call's key."""
    return _key_overrides.get().get(call_id, call_id)


@contextmanager
def bind_idempotency_keys(keys: Mapping[str, str]) -> Iterator[None]:
    """Run the block with `keys` (call id -> idempotency key) in force, so
    tool calls started inside it, including in tasks it spawns, use them."""
    token = _key_overrides.set(MappingProxyType(dict(keys)))
    try:
        yield
    finally:
        _key_overrides.reset(token)


@contextmanager
def bind_tool_call(context: ToolCallContext) -> Iterator[None]:
    """Make `context` the current tool call for the duration of the block."""
    token = _current.set(context)
    try:
        yield
    finally:
        _current.reset(token)
