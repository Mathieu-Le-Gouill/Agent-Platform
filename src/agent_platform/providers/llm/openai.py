from typing import AsyncIterator

from openai import AsyncOpenAI

from bridges.generation.openai import to_openai_params
from bridges.message.openai import from_openai_message, to_openai_message

from models.generation import GenerationConfig
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
        config: GenerationConfig | None = None,
    ) -> AssistantMessage | None:

        response = await self.client.chat.completions.create(
            model=model,
            messages=[to_openai_message(m) for m in prompt.messages],
            **to_openai_params(config),
        )

        if not response.choices:
            return None

        message = response.choices[0].message
        return from_openai_message(message)
    

    async def stream(
        self,
        prompt: Prompt,
        model: str,
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[str]:

        stream = await self.client.chat.completions.create(
            model=model,
            messages=[to_openai_message(m) for m in prompt.messages],
            **to_openai_params(config),
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
        config: GenerationConfig | None = None,
    ) -> AssistantMessage | None:

        response = await self.client.chat.completions.create(
            model=model,
            messages=[to_openai_message(m) for m in prompt.messages],
            **to_openai_params(config),
        )

        if not response.choices:
            return None

        message = response.choices[0].message
        return from_openai_message(message)