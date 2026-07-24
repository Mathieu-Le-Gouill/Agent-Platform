from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from agent_platform.core.schemas.enums import FinishReason
from agent_platform.core.schemas.message import AssistantMessage
from agent_platform.core.schemas.token import TokenUsage


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
