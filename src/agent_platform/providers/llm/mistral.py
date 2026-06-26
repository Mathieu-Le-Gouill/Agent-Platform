from mistralai.client import Mistral
from typing import AsyncIterator

from bridges.message.mistral import from_mistral_message, to_mistral_message
from bridges.generation.mistral import to_mistral_params
from models.message import AssistantMessage
from models.generation import GenerationConfig
from models.prompt import Prompt


class MistralLLM:
    client: Mistral


    def __init__(self, api_key: str) -> None:
        self.client = Mistral(api_key=api_key)


    async def generate(
        self,
        prompt: Prompt,
        model: str,
        config: GenerationConfig | None = None,
    ) -> AssistantMessage | None:

        response = await self.client.chat.complete_async(
            model=model,
            messages=[to_mistral_message(m) for m in prompt.messages],
            **to_mistral_params(config),
        )
        
        if not response.choices:
            return None

        message = response.choices[0].message
        return from_mistral_message(message) if message is not None else None

    
    async def stream(
        self,
        prompt: Prompt,
        model: str,
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[str]:

        response = await self.client.chat.stream_async(
            model=model,
            messages=[to_mistral_message(m) for m in prompt.messages],
            **to_mistral_params(config),
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
        config: GenerationConfig | None = None,
    ) -> AssistantMessage | None:

        response = await self.client.chat.complete_async(
            model=model,
            messages=[to_mistral_message(m) for m in prompt.messages],
            **to_mistral_params(config),
        )

        if not response.choices:
            return None

        message = response.choices[0].message
        return from_mistral_message(message) if message is not None else None
    


