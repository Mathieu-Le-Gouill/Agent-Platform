from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from providers.llm.message import AssistantMessage
from models.token import TokenUsage

class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALL = "tool_call"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass(slots=True, frozen=True)
class LLMResponse:
    message: AssistantMessage
    usage: TokenUsage
    model: str
    latency_ms: float | None = None
    finish_reason: FinishReason = FinishReason.STOP


@dataclass(slots=True, frozen=True)
class StreamChunk:
    delta: str
    finish_reason: FinishReason | None = None
    usage: TokenUsage | None = None
    #tool_call_delta: ToolCallDelta | None = None