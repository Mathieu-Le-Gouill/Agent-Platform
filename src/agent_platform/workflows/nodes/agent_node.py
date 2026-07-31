from __future__ import annotations

from typing import TypeVar

from agent_platform.agents.agent import Agent
from agent_platform.agents.executor import AgentExecutor
from agent_platform.workflows.graph import NodeFn
from agent_platform.workflows.state import MessagesState

__all__ = ["agent_node"]

StateT = TypeVar("StateT", bound=MessagesState)


def agent_node(agent: Agent, *, max_iterations: int = 10) -> NodeFn[StateT]:
    """Adapt `agent` into a workflow node.

    Runs `agent`'s think-act loop (via a private `AgentExecutor`, the same
    loop `ConversationAgent` uses) over `state.messages`, then returns a new
    state with the accumulated turn written back. No workflow-specific agent
    logic exists here; this is purely an adapter, so a hardened `Agent`
    (guardrails, `response_schema`, tool validation/retry) behaves identically
    whether driven directly or through a workflow node.
    """
    executor = AgentExecutor(agent, max_iterations=max_iterations)

    async def node(state: StateT) -> StateT:
        _, messages = await executor.run_with_messages(list(state.messages))
        return state.model_copy(update={"messages": messages})

    return node
