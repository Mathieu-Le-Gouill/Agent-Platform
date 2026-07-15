from __future__ import annotations

from uuid import UUID, uuid4
from typing import Any

from pydantic import BaseModel, Field

from agent_platform.core.schemas.enums import Language


class Utterance(BaseModel, frozen=True):
    speaker: str | None = None
    text: str = ""
    start_ms: int | None = None
    end_ms: int | None = None
    confidence: float | None = None


class Transcript(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    utterances: list[Utterance] = Field(default_factory=list)
    language: Language | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
