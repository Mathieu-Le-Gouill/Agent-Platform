from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from agent_platform.core.interfaces.translation.config import TranslationConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.enums import Language

ConfigT = TypeVar("ConfigT", bound=TranslationConfig)


class BaseTranslator(ABC, Generic[ConfigT]):
    @abstractmethod
    def translate(
        self,
        content: TextChunk,
        target: Language,
        source: Language | None = None,
        config: ConfigT | None = None,
    ) -> TextChunk: ...

    @abstractmethod
    async def atranslate(
        self,
        content: TextChunk,
        target: Language,
        source: Language | None = None,
        config: ConfigT | None = None,
    ) -> TextChunk: ...
