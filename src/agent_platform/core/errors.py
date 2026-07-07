class PlatformError(Exception):
    pass


class ProviderError(PlatformError):
    pass


class ConfigError(PlatformError):
    pass


class NotFoundError(PlatformError):
    pass


class ValidationError(PlatformError):
    pass


# --- LLM errors (kept for backward compat) ---


class LLMError(PlatformError):
    pass


class LLMGenerationError(LLMError):
    pass


class LLMTimeoutError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    pass


# --- Agent errors ---


class AgentError(PlatformError):
    pass


class AgentThinkError(AgentError):
    pass


class AgentActError(AgentError):
    pass


class AgentMaxIterations(AgentError):
    pass
