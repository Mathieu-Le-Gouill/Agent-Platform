from typing import AsyncIterator

from anthropic import AsyncAnthropic

from bridges.message.anthropic import extract_system, from_anthropic_message, to_anthropic_message
from bridges.generation.anthropic import to_anthropic_params

from models.message import AssistantMessage
from models.prompt import Prompt
from models.generation import GenerationConfig


class AnthropicLLM:
    client: AsyncAnthropic


    def __init__(self, api_key: str) -> None:
        self.client = AsyncAnthropic(api_key=api_key)


    async def generate(
        self,
        prompt: Prompt,
        model: str,
        config: GenerationConfig | None = None,
    ) -> AssistantMessage | None:

        system, messages = extract_system(prompt)

        response = await self.client.messages.create(
            model=model,
            system=system or "",
            messages=[to_anthropic_message(m) for m in messages],
            **to_anthropic_params(config),
        )

        return from_anthropic_message(response)
    

    async def stream(
        self,
        prompt: Prompt,
        model: str,
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[str]:

        system, messages = extract_system(prompt)

        async with self.client.messages.stream(
            model=model,
            system=system or "",
            messages=[to_anthropic_message(m) for m in messages],
            **to_anthropic_params(config),
        ) as stream:
            async for text in stream.text_stream:
                yield text
    

    async def chat(
        self,
        prompt: Prompt,
        model: str,
        config: GenerationConfig | None = None,
    ) -> AssistantMessage | None:
        
        system, messages = extract_system(prompt)

        response = await self.client.messages.create(
            model=model,
            system=system or "",
            messages=[to_anthropic_message(m) for m in messages],
            **to_anthropic_params(config),
        )

        return from_anthropic_message(response)