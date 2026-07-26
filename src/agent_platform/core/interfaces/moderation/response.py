from __future__ import annotations

from pydantic import BaseModel

from agent_platform.core.schemas.score import Score


class ModerationCategory(BaseModel, frozen=True):
    name: str
    flagged: bool
    score: Score


class ModerationResult(BaseModel, frozen=True):
    flagged: bool
    categories: list[ModerationCategory]
