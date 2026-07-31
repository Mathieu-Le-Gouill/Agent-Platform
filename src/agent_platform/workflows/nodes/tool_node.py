from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar
from uuid import uuid4

from agent_platform.agents.tools.base import Tool
from agent_platform.agents.tools.registry import ToolRegistry
from agent_platform.core.schemas.message import ToolCall, ToolMessage
from agent_platform.workflows.graph import NodeFn
from agent_platform.workflows.state import WorkflowState

__all__ = ["tool_node"]

StateT = TypeVar("StateT", bound=WorkflowState)


def tool_node(
    tool: Tool,
    *,
    input_fn: Callable[[StateT], dict[str, Any]],
    output_fn: Callable[[StateT, ToolMessage], StateT],
) -> NodeFn[StateT]:
    """Adapt `tool` into a deterministic workflow node: no LLM call, just a
    direct, always-taken invocation of `tool` as one step in the graph.

    `input_fn` extracts the tool's keyword arguments from `state`; `output_fn`
    folds the resulting `ToolMessage` back into a new state. Runs through a
    private single-tool `ToolRegistry` so the same argument
    validation/timeout/error-wrapping (`resolve_call`/`call_and_wrap`) a tool
    gets inside `Agent.act()` applies here too, instead of calling
    `tool.run()` directly and re-implementing that handling.
    """
    registry = ToolRegistry()
    registry.register(tool)

    async def node(state: StateT) -> StateT:
        call = ToolCall(id=str(uuid4()), name=tool.name, arguments=input_fn(state))
        message = await registry.call_and_wrap(call)
        return output_fn(state, message)

    return node
