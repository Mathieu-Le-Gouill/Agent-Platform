from typing import Protocol

class BaseLLM(Protocol):
    async def generate(
        self,
        prompt: str,
        model: str,
    ) -> str | None: 
        ...


    async def stream(
        self,
        prompt: str,
        model: str,
    ) -> str | None: 
        ...


    async def chat(
        self,
        prompt: str,
        model: str,
    ) -> str | None: 
        ...