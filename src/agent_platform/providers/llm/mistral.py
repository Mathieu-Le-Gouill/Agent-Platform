from mistralai.client import Mistral
from typing import AsyncIterator

from bridges.message.mistral import from_mistral, to_mistral
from models.message import AssistantMessage
from models.prompt import Prompt


class MistralLLM:
    client: Mistral


    def __init__(self, api_key: str) -> None:
        self.client = Mistral(api_key=api_key)


    async def generate(
        self,
        prompt: Prompt,
        model: str,
    ) -> AssistantMessage | None:
        """Single-turn generation, no conversation history, just the prompt."""

        response = await self.client.chat.complete_async(
            model=model,
            messages=[to_mistral(m) for m in prompt.messages],
        )
        
        if not response.choices:
            return None

        message = response.choices[0].message
        return from_mistral(message) if message is not None else None

    
    async def stream(
        self,
        prompt: Prompt,
        model: str,
    ) -> AsyncIterator[str]:
        """
        Streams the assistant reply token by token.

        Usage:
            async for token in llm.stream(prompt, model):
                print(token, end="", flush=True)
        """

        response = await self.client.chat.stream_async(
            model=model,
            messages=[to_mistral(m) for m in prompt.messages],
        )

        async for event in response:
            delta = event.data.choices[0].delta
            token = getattr(delta, "content", None)
            if isinstance(token, str):
                yield token


    async def chat(
        self,
        prompt: Prompt,
        model: str,
    ) -> AssistantMessage | None:
        """Multi-turn chat, full conversation history passed via Prompt."""

        response = await self.client.chat.complete_async(
            model=model,
            messages=[to_mistral(m) for m in prompt.messages],
        )

        if not response.choices:
            return None

        message = response.choices[0].message
        return from_mistral(message) if message is not None else None
    


