from __future__ import annotations

from agent_platform.core.errors import PlatformError

__all__ = ["WorkflowError", "WorkflowValidationError", "WorkflowExecutionError"]


class WorkflowError(PlatformError):
    pass


class WorkflowValidationError(WorkflowError):
    """Raised by `WorkflowGraph.compile()` when the graph is malformed."""


class WorkflowExecutionError(WorkflowError):
    """Raised by `CompiledWorkflow` at run time (an unresolvable router
    destination, or a checkpointer used without a `run_id`)."""
