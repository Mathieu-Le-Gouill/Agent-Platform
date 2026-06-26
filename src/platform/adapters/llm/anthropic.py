from typing import AsyncIterator

from anthropic import AsyncAnthropic

from mappers.anthropic.generation import to_anthropic as to_anthropic_params
from mappers.anthropic.message import extract_system, to_anthropic as to_anthropic_message, from_anthropic as from_anthropic_message

from adapters.llm.message import AssistantMessage
from adapters.llm.prompt import Prompt
from adapters.llm.config import GenerationConfig


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