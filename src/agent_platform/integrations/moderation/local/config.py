from __future__ import annotations

from pydantic import Field

from agent_platform.core.interfaces.moderation.config import ModerationConfig

_DEFAULT_RULES: dict[str, list[str]] = {
    "violence": ["kill you", "murder", "bomb"],
    "self-harm": ["suicide", "self-harm", "self harm"],
    "hate": ["slur"],
}


class LocalModerationConfig(ModerationConfig):
    # Category name -> substrings that flag it (case-insensitive). Deliberately
    # simple: this is an offline fallback, not a replacement for a real
    # classifier-backed moderation provider.
    rules: dict[str, list[str]] = Field(default_factory=lambda: dict(_DEFAULT_RULES))
