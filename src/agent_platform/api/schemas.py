from __future__ import annotations

from pydantic import BaseModel

from agent_platform.core.schemas.token import TokenUsage


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    usage: TokenUsage
    estimated_cost: float | None = None
