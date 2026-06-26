from typing import AsyncIterator

from ollama import AsyncClient

from bridges.message.ollama import from_ollama, to_ollama
from models.message import AssistantMessage
from models.prompt import Prompt


class OllamaLLM:
    client: AsyncClient


    def __init__(self, host: str) -> None:
        self.client = AsyncClient(host=host)


    async def generate(
        self,
        prompt: Prompt,
        model: str,
    ) -> AssistantMessage | None:
        """Single-turn generation."""

        response = await self.client.chat(
            model=model,
            messages=[to_ollama(m) for m in prompt.messages],
        )

        message = response.get("message")

        if message is None:
            return None

        return from_ollama(message)
    

    async def stream(
        self,
        prompt: Prompt,
        model: str,
    ) -> AsyncIterator[str]:
        """Stream assistant tokens."""

        stream = await self.client.chat(
            model=model,
            messages=[to_ollama(m) for m in prompt.messages],
            stream=True,
        )

        async for chunk in stream:
            message = chunk.get("message", {})
            content = message.get("content")

            if content:
                yield content


    async def chat(
        self,
        prompt: Prompt,
        model: str,
    ) -> AssistantMessage | None:
        """Multi-turn chat."""

        response = await self.client.chat(
            model=model,
            messages=[to_ollama(m) for m in prompt.messages],
        )

        message = response.get("message")

        if message is None:
            return None

        return from_ollama(message)