from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

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


class ToolCallDelta(BaseModel, frozen=True):
    """One fragment of a tool call arriving mid-stream.

    `id`/`name` arrive once per call (vendors differ on which chunk carries
    them); `arguments_delta` is a fragment of the arguments JSON string to be
    concatenated by index and parsed once the call is complete. A vendor whose
    streaming API returns tool calls atomically (no argument fragmentation)
    emits a single delta per call with `arguments_delta` set to the full JSON.
    """

    index: int
    id: str | None = None
    name: str | None = None
    arguments_delta: str | None = None


class StreamChunk(BaseModel, frozen=True):
    delta: str
    finish_reason: FinishReason = FinishReason.STOP
    usage: TokenUsage | None = None
    tool_call_deltas: list[ToolCallDelta] = Field(default_factory=list)
