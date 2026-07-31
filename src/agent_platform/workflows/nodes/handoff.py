from __future__ import annotations

from typing import TypeVar

from agent_platform.agents.agent import Agent
from agent_platform.workflows.graph import NodeFn
from agent_platform.workflows.nodes.agent_node import agent_node
from agent_platform.workflows.state import HandoffState

__all__ = ["handoff_node"]

StateT = TypeVar("StateT", bound=HandoffState)


def handoff_node(agent: Agent, *, max_iterations: int = 10) -> NodeFn[StateT]:
    """Hand control to `agent`, sharing the accumulated `state.messages`.

    Built directly on `agent_node`: a "handoff" is that same adapter, plus
    stamping `state.active_agent` afterward so downstream nodes or
    observability can tell which agent produced a given turn in a multi-agent
    workflow. Two agents can be swapped between mid-run purely by wiring
    different `handoff_node(agent)` instances into the graph; no separate
    delegation protocol exists between agents beyond the shared state.
    """
    run_agent: NodeFn[StateT] = agent_node(agent, max_iterations=max_iterations)

    async def node(state: StateT) -> StateT:
        new_state = await run_agent(state)
        return new_state.model_copy(update={"active_agent": agent.name})

    return node
