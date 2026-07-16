from __future__ import annotations

from agent_platform.core.errors import PlatformError

__all__ = [
    "ToolError",
    "ToolNotFoundError",
    "ToolRegistrationError",
]


class ToolError(PlatformError):
    pass


class ToolNotFoundError(ToolError):
    pass


class ToolRegistrationError(ToolError):
    pass
