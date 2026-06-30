from typing import Protocol
from models.protocols.text_unit import TextUnit
from platform.models.enums.language import Language


class BaseTranslator(Protocol):
    
    async def translate(
        self,
        content: TextUnit,
        target: Language,
        source: Language | None = None
    ) -> TextUnit: 
        ...