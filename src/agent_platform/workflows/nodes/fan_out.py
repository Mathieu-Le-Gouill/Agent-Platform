from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import TypeVar

from agent_platform.core.observability import WorkflowNode, observe
from agent_platform.workflows.graph import NodeFn, bind_step, current_step
from agent_platform.workflows.state import WorkflowState

__all__ = ["fan_out"]

StateT = TypeVar("StateT", bound=WorkflowState)


def fan_out(
    branches: Sequence[tuple[str, NodeFn[StateT]]],
    merge: Callable[[StateT, list[StateT]], StateT],
) -> NodeFn[StateT]:
    """Run `branches` concurrently over independent copies of the incoming
    state, then combine their results with `merge`.

    Built as a plain `NodeFn` combinator, the same way `handoff_node` is
    built on top of `agent_node`, rather than the engine growing a
    first-class "parallel" node type. Each branch gets its own
    `workflow_node` span so a fan-out step is as observable as any other
    node in the graph.
    """

    async def node(state: StateT) -> StateT:
        parent = current_step()

        async def run_branch(name: str, branch: NodeFn[StateT]) -> StateT:
            with observe(WorkflowNode(node=name)):
                if parent is None:
                    return await branch(state)
                # A context of its own per branch: the branches share this
                # node's run and step, so the name tells their tool calls apart.
                with bind_step(parent.for_branch(name)):
                    return await branch(state)

        results = await asyncio.gather(
            *(run_branch(name, branch) for name, branch in branches)
        )
        return merge(state, list(results))

    return node
