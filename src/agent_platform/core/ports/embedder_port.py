from typing import Protocol

class EmbedderPort(Protocol):
    async def embed(self, data: str) -> str: 
        ...