from __future__ import annotations

from agent_platform.core.interfaces.llm.response import ResponseFormat
from agent_platform.core.schemas.config import ProviderConfig


class GenerationConfig(ProviderConfig):
    model: str = ""
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] = []
    seed: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    timeout: float | None = None
    max_retries: int | None = None
    response_format: ResponseFormat = ResponseFormat.TEXT
    json_schema: dict | None = None
