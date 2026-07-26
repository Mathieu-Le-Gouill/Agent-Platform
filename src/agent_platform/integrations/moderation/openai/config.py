from __future__ import annotations

from agent_platform.core.interfaces.moderation.config import ModerationConfig


class OpenAIModerationConfig(ModerationConfig):
    model: str = "omni-moderation-latest"
