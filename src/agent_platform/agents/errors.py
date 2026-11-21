from __future__ import annotations

from agent_platform.core.errors import PlatformError
from agent_platform.core.schemas.message import AssistantMessage

__all__ = [
    "AgentError",
    "AgentThinkError",
    "AgentActError",
    "AgentMaxIterations",
    "AgentRecoveryExhausted",
    "AgentGuardrailError",
]


class AgentError(PlatformError):
    pass


class AgentThinkError(AgentError):
    pass


class AgentActError(AgentError):
    pass


class AgentMaxIterations(AgentError):
    """Raised when the loop used every iteration without a final answer."""

    def __init__(self, message: str = "") -> None:
        super().__init__(message, code="agent_max_iterations")


class AgentRecoveryExhausted(AgentError):
    """Raised when the corrective retry of a tool call or a schema still fails.

    Deliberately not a `RecoveryExhausted`: the executor catches that one to
    handle its own retry, and must not catch the agent's error.
    `last_message` is the last invalid answer, kept for diagnosis only.
    """

    def __init__(
        self, message: str = "", *, last_message: AssistantMessage | None = None
    ) -> None:
        super().__init__(message, code="agent_recovery_exhausted")
        self.last_message = last_message


class AgentGuardrailError(AgentError):
    pass
