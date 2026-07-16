from __future__ import annotations

from agent_platform.core.errors import PlatformError

__all__ = [
    "AgentError",
    "AgentThinkError",
    "AgentActError",
    "AgentMaxIterations",
]


class AgentError(PlatformError):
    pass


class AgentThinkError(AgentError):
    pass


class AgentActError(AgentError):
    pass


class AgentMaxIterations(AgentError):
    pass
