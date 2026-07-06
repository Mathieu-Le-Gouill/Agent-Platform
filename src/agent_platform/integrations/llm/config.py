from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ResponseFormat(Enum):
    TEXT = "text"
    JSON = "json"
    JSON_SCHEMA = "json_schema"


@dataclass(slots=True, frozen=True)
class GenerationConfig:
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] = field(default_factory=list)
    seed: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    timeout: float | None = None
    max_retries: int = 3
    response_format: ResponseFormat = ResponseFormat.TEXT
    json_schema: dict | None = None


@dataclass(slots=True, frozen=True)
class AnthropicConfig(GenerationConfig):
    thinking: bool = False
    thinking_budget: int | None = None
    cache_control: bool = False


@dataclass(slots=True, frozen=True)
class OpenAIConfig(GenerationConfig):
    reasoning_effort: str | None = None  # for o1/o3
    parallel_tool_calls: bool = True
