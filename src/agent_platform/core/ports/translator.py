from typing import Protocol

class TranslatorPort(Protocol):
    async def translate(
        self,
        content: str,
    ) -> str: 
        ...