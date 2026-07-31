from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from agent_platform.workflows.graph import Router
from agent_platform.workflows.state import WorkflowState

__all__ = ["TRUE", "FALSE", "binary_router"]

StateT = TypeVar("StateT", bound=WorkflowState)

TRUE = "true"
FALSE = "false"


def binary_router(predicate: Callable[[StateT], bool]) -> Router[StateT]:
    """Wrap a boolean `predicate` into a `Router`, for the common case of a
    yes/no conditional edge:

    ```python
    graph.add_conditional_edges(
        "check", binary_router(lambda s: s.needs_review), {TRUE: "review", FALSE: END}
    )
    ```
    """

    def router(state: StateT) -> str:
        return TRUE if predicate(state) else FALSE

    return router
