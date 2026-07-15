from __future__ import annotations

from agent_platform.core.errors import PlatformError

__all__ = [
    "LLMError",
    "LLMGenerationError",
    "LLMTimeoutError",
    "LLMRateLimitError",
    "LLMContextWindowError",
]


class LLMError(PlatformError):
    pass


class LLMGenerationError(LLMError):
    pass


class LLMTimeoutError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    pass


class LLMContextWindowError(LLMError):
    pass
