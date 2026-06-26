from typing import AsyncIterator

from ollama import AsyncClient

from bridges.message.ollama import from_ollama_message, to_ollama_message
from bridges.generation.ollama import to_ollama_params

from models.message import AssistantMessage
from models.generation import GenerationConfig
from models.prompt import Prompt


class OllamaLLM:
    client: AsyncClient


    def __init__(self, host: str) -> None:
        self.client = AsyncClient(host=host)


    async def generate(
        self,
        prompt: Prompt,
        model: str,
        config: GenerationConfig | None = None,
    ) -> AssistantMessage | None:

        response = await self.client.chat(
            model=model,
            messages=[to_ollama_message(m) for m in prompt.messages],
            **to_ollama_params(config),
        )

        message = response.get("message")

        if message is None:
            return None

        return from_ollama_message(message)
    

    async def stream(
        self,
        prompt: Prompt,
        model: str,
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[str]:

        stream = await self.client.chat(
            model=model,
            messages=[to_ollama_message(m) for m in prompt.messages],
            stream=True,
            **to_ollama_params(config),
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
        config: GenerationConfig | None = None,
    ) -> AssistantMessage | None:

        response = await self.client.chat(
            model=model,
            messages=[to_ollama_message(m) for m in prompt.messages],
            **to_ollama_params(config),
        )

        message = response.get("message")

        if message is None:
            return None

        return from_ollama_message(message)