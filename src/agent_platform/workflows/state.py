from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

from agent_platform.core.schemas.message import Message

__all__ = ["WorkflowState", "MessagesState", "HandoffState", "WorkflowCheckpoint"]


class WorkflowState(BaseModel):
    """Base class for a workflow's typed state.

    Subclass per workflow with whatever fields its nodes read/write. State
    flows through the graph by value: each node returns a new/updated
    instance (typically via `state.model_copy(update={...})`), the same
    "state in, state out" shape modern graph-orchestration libraries use, kept
    here as a plain Pydantic model rather than a framework-specific TypedDict.
    """


class MessagesState(WorkflowState):
    """Ready-made state for message-driven workflows.

    The shape `workflows/nodes/agent_node.py` and `workflows/nodes/handoff.py`
    require: they read `messages`, run an `Agent`'s think-act loop over it,
    and write the accumulated `list[Message]` back.
    """

    messages: list[Message] = Field(default_factory=list)


class HandoffState(MessagesState):
    """`MessagesState` plus which agent produced the latest turn.

    `workflows/nodes/handoff.py::handoff_node` stamps `active_agent` after
    each turn so downstream nodes/observability can tell which agent handled
    a given step in a multi-agent workflow.
    """

    active_agent: str = ""


StateT = TypeVar("StateT", bound=WorkflowState)


class WorkflowCheckpoint(BaseModel, Generic[StateT]):
    """What `CompiledWorkflow.arun`/`resume` persist after each node.

    Captures the state at that point *and* which node runs next, so a
    resumed run knows where to continue rather than only what was last
    computed (a plain `Checkpointer[StateT]` recording only `state` couldn't
    reconstruct control flow after a conditional edge).
    """

    next_node: str
    state: StateT
