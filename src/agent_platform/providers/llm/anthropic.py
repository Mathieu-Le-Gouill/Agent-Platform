from typing import AsyncIterator, Optional

from anthropic import AsyncAnthropic

from bridges.message.anthropic import extract_system, from_anthropic, to_anthropic
from models.message import AssistantMessage
from models.prompt import Prompt


class AnthropicLLM:
    client: AsyncAnthropic


    def __init__(self, api_key: str) -> None:
        self.client = AsyncAnthropic(api_key=api_key)


    async def generate(
        self,
        prompt: Prompt,
        model: str,
        max_tokens: int = 1024,
    ) -> Optional[AssistantMessage]:
        """Single-turn generation — stateless, no conversation history."""
        system, messages = extract_system(prompt)

        response = await self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system or "",
            messages=[to_anthropic(m) for m in messages],
        )

        return from_anthropic(response)
    

    async def stream(
        self,
        prompt: Prompt,
        model: str,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """
        Streams the assistant reply token by token.

        Usage:
            async for token in llm.stream(prompt, model):
                print(token, end="", flush=True)
        """
        system, messages = extract_system(prompt)

        async with self.client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system or "",
            messages=[to_anthropic(m) for m in messages],
        ) as stream:
            async for text in stream.text_stream:
                yield text
    

    async def chat(
        self,
        prompt: Prompt,
        model: str,
        max_tokens: int = 1024,
    ) -> Optional[AssistantMessage]:
        """Multi-turn chat — full conversation history passed via Prompt."""
        system, messages = extract_system(prompt)

        response = await self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system or "",
            messages=[to_anthropic(m) for m in messages],
        )

        return from_anthropic(response)