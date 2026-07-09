from __future__ import annotations

from pydantic import BaseModel
from enum import Enum

from agent_platform.models.message import AssistantMessage
from agent_platform.models.token import TokenUsage
from agent_platform.models.enums import FinishReason


class ResponseFormat(Enum):
    TEXT = "text"
    JSON = "json"
    JSON_SCHEMA = "json_schema"

class LLMResponse(BaseModel, frozen=True):
    message: AssistantMessage | None
    usage: TokenUsage
    model: str
    latency_ms: float | None = None
    finish_reason: FinishReason = FinishReason.STOP


class StreamChunk(BaseModel, frozen=True):
    delta: str
    finish_reason: FinishReason = FinishReason.STOP
    usage: TokenUsage | None = None
