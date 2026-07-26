from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENT_PLATFORM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # "<provider>:<model>" strings (see core/config.py::parse_model_string), resolved
    # by config/container.py::build_provider_from_model_string against each domain's
    # `PROVIDER_ALIASES` map (e.g. "openai:gpt-4o-mini", "anthropic:claude-sonnet-4-5").
    default_llm_model: str = "openai:gpt-4o-mini"
    default_image_model: str = "dalle:dall-e-3"
    default_audio_model: str = "openai:whisper-1"
    agent_name: str = "assistant"
    agent_system_prompt: str | None = None
    max_iterations: int = 10
    log_level: str = "INFO"
    api_host: str = "127.0.0.1"
    api_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()


__all__ = ["Settings", "get_settings"]
