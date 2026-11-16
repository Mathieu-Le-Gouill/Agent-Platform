from __future__ import annotations

from agent_platform.core.errors import PlatformError

__all__ = [
    "ToolError",
    "ToolNotFoundError",
    "ToolRegistrationError",
    "ToolCallValidationError",
    "ToolTimeoutError",
    "ToolSourceNotAllowedError",
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


class ToolSourceNotAllowedError(ToolError):
    """A tool refused a file path or URL its policy does not allow."""
