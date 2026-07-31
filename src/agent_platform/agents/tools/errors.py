from __future__ import annotations

from agent_platform.core.errors import PlatformError

__all__ = [
    "ToolError",
    "ToolNotFoundError",
    "ToolRegistrationError",
    "ToolCallValidationError",
    "ToolTimeoutError",
]


class ToolError(PlatformError):
    pass


class ToolNotFoundError(ToolError):
    pass


class ToolRegistrationError(ToolError):
    pass


class ToolCallValidationError(ToolError):
    pass


class ToolTimeoutError(ToolError):
    pass
