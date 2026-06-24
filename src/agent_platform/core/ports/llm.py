from typing import Protocol

class LLMPort(Protocol):
    async def generate(
        self,
        prompt: str,
        model: str,
    ) -> str | None: 
        ...