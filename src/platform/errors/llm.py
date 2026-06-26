class LLMError(Exception):
    """Base exception for LLM operations."""


class LLMGenerationError(LLMError):
    """Raised when generation fails."""


class LLMTimeoutError(LLMError):
    """Raised when an LLM request times out."""


class LLMRateLimitError(LLMError):
    """Raised when the backend rate limits the request."""