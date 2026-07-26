from __future__ import annotations

from agent_platform.core.interfaces.moderation.base import BaseModerationProvider
from agent_platform.core.interfaces.moderation.response import (
    ModerationCategory,
    ModerationResult,
)
from agent_platform.core.schemas.score import Score
from agent_platform.integrations.moderation.local.config import LocalModerationConfig


class LocalModeration(BaseModerationProvider[LocalModerationConfig]):
    """Offline keyword-rule fallback: no credentials, no network call.

    Both `moderate`/`amoderate` are required by the ABC, but keyword matching
    is instantaneous, so `amoderate` just calls `moderate` directly (same
    "instant local compute" reasoning as `TransformersClassifier`, minus any
    blocking model load that would need `asyncio.to_thread`).
    """

    def _default_config(self) -> LocalModerationConfig:
        return LocalModerationConfig()

    def moderate(
        self,
        text: str,
        config: LocalModerationConfig | None = None,
    ) -> ModerationResult:
        config = config or self._default_config()
        lowered = text.lower()

        categories = []
        for name, patterns in config.rules.items():
            flagged = any(pattern.lower() in lowered for pattern in patterns)
            categories.append(
                ModerationCategory(
                    name=name,
                    flagged=flagged,
                    score=Score.confidence(1.0 if flagged else 0.0),
                )
            )

        return ModerationResult(
            flagged=any(c.flagged for c in categories),
            categories=categories,
        )

    async def amoderate(
        self,
        text: str,
        config: LocalModerationConfig | None = None,
    ) -> ModerationResult:
        return self.moderate(text, config)
