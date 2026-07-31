from __future__ import annotations

from typing import Any

from agent_platform.core.errors import PlatformError

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
    pass


class AgentRecoveryExhausted(AgentError):
    """Raised when a single corrective retry (bad tool call or schema mismatch) still fails."""

    def __init__(self, message: str = "", *, last_result: Any = None) -> None:
        super().__init__(message)
        self.last_result = last_result


class AgentGuardrailError(AgentError):
    pass
