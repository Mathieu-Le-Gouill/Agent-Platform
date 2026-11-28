from __future__ import annotations

import json
import logging
from typing import Any

from agent_platform.core.interfaces.llm.response import ToolCallDelta
from agent_platform.core.schemas.message import ToolCall

__all__ = ["assemble_tool_calls", "tool_call_from_json"]

logger = logging.getLogger(__name__)


def tool_call_from_json(call_id: str, name: str, raw_arguments: str) -> ToolCall:
    """Build a `ToolCall` from the model's raw arguments text without raising.

    Arguments that are not a JSON object set `arguments_error` instead, so the
    registry rejects the call and the model gets a Corrective Retry rather
    than the turn failing.
    """
    arguments: dict[str, Any] = {}
    arguments_error: str | None = None
    try:
        decoded = json.loads(raw_arguments) if raw_arguments else {}
    except json.JSONDecodeError as exc:
        # `exc` describes the syntax error only, never the argument text.
        arguments_error = str(exc)
    else:
        if isinstance(decoded, dict):
            arguments = decoded
        else:
            arguments_error = "arguments must be a JSON object"
    if arguments_error is not None:
        logger.warning("Tool call '%s' has arguments that are not a JSON object", name)
    return ToolCall(
        id=call_id, name=name, arguments=arguments, arguments_error=arguments_error
    )


def assemble_tool_calls(deltas: list[ToolCallDelta]) -> list[ToolCall]:
    """Join streamed `ToolCallDelta`s by index into complete `ToolCall`s, in
    the order each index first appears."""
    ids: dict[int, str] = {}
    names: dict[int, str] = {}
    fragments: dict[int, list[str]] = {}
    for delta in deltas:
        fragments.setdefault(delta.index, [])
        if delta.id is not None:
            ids[delta.index] = delta.id
        if delta.name is not None:
            names[delta.index] = delta.name
        if delta.arguments_delta is not None:
            fragments[delta.index].append(delta.arguments_delta)
    return [
        tool_call_from_json(ids.get(index, ""), names.get(index, ""), "".join(parts))
        for index, parts in fragments.items()
    ]
