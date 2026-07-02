from abc import ABC, abstractmethod

from agent_platform.models.protocols.text_unit import TextUnit
from agent_platform.models.enums.language import Language


class BaseTranslator(ABC):
    
    @abstractmethod
    async def translate(
        self,
        content: TextUnit,
        target: Language,
        source: Language | None = None
    ) -> TextUnit: 
        ...