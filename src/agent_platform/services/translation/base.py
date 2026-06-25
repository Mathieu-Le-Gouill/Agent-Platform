from typing import Protocol

class BaseTranslator(Protocol):
    async def translate(
        self,
        content: str,
    ) -> str: 
        ...