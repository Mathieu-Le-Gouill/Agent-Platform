from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    llm_rate_limit: float | None = None
    llm_rate_limit_burst: float | None = None
    # MCP servers to launch over stdio and register tools from, keyed by an
    # arbitrary name; each value is `[command, *args]` (the same shape as
    # Claude Desktop/VS Code's `mcpServers` config), e.g.
    # {"docs": ["npx", "-y", "@modelcontextprotocol/server-docs"]}. Empty by
    # default: no MCP tools. Only takes effect via `build_agent_async` (see
    # config/container.py), since MCP discovery is async; the plain sync
    # `build_agent` ignores this setting.
    mcp_stdio_servers: dict[str, list[str]] = Field(default_factory=dict)
    default_image_model: str = "dalle:dall-e-3"
    default_audio_model: str = "openai:whisper-1"
    agent_name: str = "assistant"
    agent_system_prompt: str | None = None
    max_iterations: int = 10
    # Most recent user turns an agent keeps in a conversation's history; older
    # turns are dropped so a long conversation stays within the model's context.
    max_history_turns: int = Field(default=20, ge=1)
    log_level: str = "INFO"
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

    @field_validator("api_keys")
    @classmethod
    def _reject_blank_api_keys(cls, keys: list[SecretStr]) -> list[SecretStr]:
        if any(not key.get_secret_value().strip() for key in keys):
            raise ValueError("api_keys must not contain an empty or blank key")
        return keys


@lru_cache
def get_settings() -> Settings:
    return Settings()


__all__ = ["Settings", "get_settings"]
