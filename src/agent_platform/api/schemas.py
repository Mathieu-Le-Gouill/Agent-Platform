from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel

from agent_platform.core.schemas.token import TokenUsage


class ChatRequest(BaseModel):
    message: str
    # `/chat` continues this conversation when set and starts a new one otherwise;
    # `/chat/stream` ignores it.
    conversation_id: UUID | None = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: UUID
    usage: TokenUsage
    estimated_cost: float | None = None
