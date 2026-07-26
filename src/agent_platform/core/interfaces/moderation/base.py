from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from agent_platform.core.interfaces.moderation.config import ModerationConfig
from agent_platform.core.interfaces.moderation.response import ModerationResult

ConfigT = TypeVar("ConfigT", bound=ModerationConfig)


class BaseModerationProvider(ABC, Generic[ConfigT]):
    """Flags text against a set of safety/policy categories.

    Follows the network-bound sync/async pairing convention (like `llm`/
    `embeddings`/`reranking`/`translation`), not `classification`'s
    async-only exemption, since the primary provider (OpenAI) is a real
    network call.
    """

    @abstractmethod
    def moderate(
        self,
        text: str,
        config: ConfigT | None = None,
    ) -> ModerationResult: ...

    @abstractmethod
    async def amoderate(
        self,
        text: str,
        config: ConfigT | None = None,
    ) -> ModerationResult: ...
