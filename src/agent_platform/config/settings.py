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

    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
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
