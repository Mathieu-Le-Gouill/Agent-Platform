from typing import AsyncIterator, Optional

from openai import AsyncOpenAI

from bridges.message.openai import from_openai, to_openai
from models.message import AssistantMessage
from models.prompt import Prompt


class OpenAILLM:
    client: AsyncOpenAI

    def __init__(self, api_key: str) -> None:
        self.client = AsyncOpenAI(api_key=api_key)

    async def generate(
        self,
        prompt: Prompt,
        model: str,
    ) -> Optional[AssistantMessage]:
        """Single-turn generation."""

        response = await self.client.chat.completions.create(
            model=model,
            messages=[to_openai(m) for m in prompt.messages],
        )

        if not response.choices:
            return None

        message = response.choices[0].message
        return from_openai(message)

    async def stream(
        self,
        prompt: Prompt,
        model: str,
    ) -> AsyncIterator[str]:
        """Stream assistant tokens."""

        stream = await self.client.chat.completions.create(
            model=model,
            messages=[to_openai(m) for m in prompt.messages],
            stream=True,
        )

        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta.content:
                yield delta.content

    async def chat(
        self,
        prompt: Prompt,
        model: str,
    ) -> Optional[AssistantMessage]:
        """Multi-turn chat."""

        response = await self.client.chat.completions.create(
            model=model,
            messages=[to_openai(m) for m in prompt.messages],
        )

        if not response.choices:
            return None

        message = response.choices[0].message
        return from_openai(message)