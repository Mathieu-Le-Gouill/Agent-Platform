from abc import ABC, abstractmethod

from agent_platform.models.chunk import TextChunk
from agent_platform.models.enums import Language


class BaseTranslator(ABC):
    @abstractmethod
    async def translate(
        self, content: TextChunk, target: Language, source: Language | None = None
    ) -> TextChunk: ...
