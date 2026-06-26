from dataclasses import dataclass, field
from enum import Enum


class ResponseFormat(Enum):
    TEXT = "text"
    JSON = "json"
    JSON_SCHEMA = "json_schema"


@dataclass
class GenerationConfig:
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float | None = None
    stop_sequences: list[str] = field(default_factory=list)
    seed: int | None = None

    response_format: ResponseFormat = ResponseFormat.TEXT
    json_schema: dict | None = None