from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, SettingsError

from agent_platform.core.errors import ConfigError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENT_PLATFORM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # "<provider>:<model>" strings (see config/model_string.py::parse_model_string), resolved
    # by config/container.py::build_provider_from_model_string against each domain's
    # `PROVIDER_ALIASES` map (e.g. "openai:gpt-4o-mini", "anthropic:claude-sonnet-4-5").
    default_llm_model: str = "openai:gpt-4o-mini"
    # Additional "<provider>:<model>" strings tried in order if default_llm_model's
    # provider's circuit breaker trips (see config/container.py::build_agent and
    # core/interfaces/llm/fallback.py::FallbackLLMProvider). Empty by default: no
    # fallback chain, the single configured provider is used directly.
    fallback_llm_models: list[str] = Field(default_factory=list)
    # Token-bucket limit applied to the fallback chain as a whole (core/resilience.py::
    # RateLimiter), shared across all entries since the budget being protected is the
    # caller's total downstream traffic, not any one vendor's. None disables limiting.
    # Only takes effect when fallback_llm_models is non-empty (see build_llm_with_fallback).
    # `llm_rate_limit` is requests/second; `llm_rate_limit_burst` is the bucket
    # size and must hold at least one request, or every call would wait forever.
    llm_rate_limit: float | None = Field(default=None, gt=0)
    llm_rate_limit_burst: float | None = Field(default=None, ge=1)
    # Provider-only generation options, keyed by LLM provider slug, as a JSON
    # object in the environment, e.g. {"anthropic": {"cache_control": true}}.
    # Each entry is validated against its provider's config class when the
    # agent is built (see config/container.py::build_llm_with_fallback). The
    # model never goes here: it comes from the "<provider>:<model>" strings.
    llm_options: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # MCP servers to launch over stdio and register tools from, keyed by an
    # arbitrary name; each value is `[command, *args]` (the same shape as
    # Claude Desktop/VS Code's `mcpServers` config), e.g.
    # {"docs": ["npx", "-y", "@modelcontextprotocol/server-docs"]}. Empty by
    # default: no MCP tools. Only takes effect via `build_agent_async` (see
    # config/container.py), since MCP discovery is async; the plain sync
    # `build_agent` ignores this setting.
    mcp_stdio_servers: dict[str, list[str]] = Field(default_factory=dict)
    # Seconds one MCP tool call may run before it is cancelled and reported as a
    # timeout, so a hung server cannot hold a request open.
    mcp_tool_timeout: float = Field(default=60.0, gt=0)
    default_image_model: str = "dalle:dall-e-3"
    default_audio_model: str = "openai:whisper-1"
    agent_name: str = "assistant"
    agent_system_prompt: str | None = None
    max_iterations: int = Field(default=10, ge=1)
    # Most recent user turns an agent keeps in a conversation's history; older
    # turns are dropped so a long conversation stays within the model's context.
    max_history_turns: int = Field(default=20, ge=1)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    # Tracing backend slug resolved against integrations/tracing's
    # PROVIDER_ALIASES by config/container.py::build_tracer, or "none" to
    # disable tracing outright. The default Langfuse backend is itself a safe
    # no-op until LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set (see
    # integrations/README.md), so this rarely needs changing.
    tracing_provider: str = "langfuse"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    # Bearer keys accepted by the API, as a JSON list in the environment
    # (`["key-1", "key-2"]`). Each key owns its own conversations. The API
    # refuses to start with none unless `api_auth_disabled` is true.
    api_keys: list[SecretStr] = Field(default_factory=list)
    # Turns authentication off: every caller is one anonymous owner. For local
    # development only.
    api_auth_disabled: bool = False
    # Conversations each key may hold in memory; creating one more evicts the
    # key's least recently used conversation.
    api_max_conversations_per_key: int = Field(default=100, ge=1)

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_case_log_level(cls, level: Any) -> Any:
        return level.upper() if isinstance(level, str) else level

    @field_validator("api_keys")
    @classmethod
    def _reject_blank_api_keys(cls, keys: list[SecretStr]) -> list[SecretStr]:
        if any(not key.get_secret_value().strip() for key in keys):
            raise ValueError("api_keys must not contain an empty or blank key")
        return keys


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, read once from the environment and `.env`.

    Raises `ConfigError` when a setting is malformed. The message names each
    bad field and its rule, never the input value, since a value may be a key.
    A failure is not cached, so the next call reads the environment again.
    """
    try:
        return Settings()
    except ValidationError as exc:
        message = _describe_validation_errors(exc)
    except SettingsError as exc:
        # Names the field and the source, never the value.
        message = str(exc)
    # Raised outside the `except` blocks so no context is attached: the
    # original exception holds the input value and would print it.
    raise ConfigError(f"Invalid settings: {message}")


def _describe_validation_errors(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
        for error in exc.errors(include_input=False, include_url=False)
    )


__all__ = ["Settings", "get_settings"]
