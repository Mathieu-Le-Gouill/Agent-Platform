from typing import AsyncIterator

from ollama import AsyncClient

from mappers.ollama.message import to_ollama as to_ollama_message, from_ollama as from_ollama_message
from mappers.ollama.generation import to_ollama as to_ollama_params

from adapters.llm.message import AssistantMessage
from adapters.llm.config import GenerationConfig
from adapters.llm.prompt import Prompt


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