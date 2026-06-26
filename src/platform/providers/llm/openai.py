from typing import AsyncIterator

from openai import AsyncOpenAI

from bridges.openai.generation import to_openai as to_openai_params
from bridges.openai.message import to_openai as to_openai_message, from_openai as from_openai_message

from providers.llm.config import GenerationConfig
from providers.llm.message import AssistantMessage
from providers.llm.prompt import Prompt


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