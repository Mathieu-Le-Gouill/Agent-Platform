from mistralai.client import Mistral
from typing import AsyncIterator

from mappers.mistral.message import to_mistral as to_mistral_message
from platform.mappers.mistral.generation_config import to_mistral as to_mistral_params

from adapters.llm.config import GenerationConfig
from adapters.llm.prompt import Prompt
from adapters.llm.response import LLMResponse, StreamChunk
from mappers.mistral.response import from_mistral_response, from_mistral_finish_reason
from models.token import TokenUsage


class MistralLLM:
    client: Mistral


    def __init__(self, api_key: str) -> None:
        self.client = Mistral(api_key=api_key)


    async def generate(
        self,
        prompt: Prompt,
        model: str,
        config: GenerationConfig | None = None,
    ) -> LLMResponse:

        response = await self.client.chat.complete_async(
            model=model,
            messages=[to_mistral_message(m) for m in prompt.messages],
            **to_mistral_params(config),
        )
        
        return from_mistral_response(response)

    
    async def stream(
        self,
        prompt: Prompt,
        model: str,
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[StreamChunk]:

        response = await self.client.chat.stream_async(
            model=model,
            messages=[to_mistral_message(m) for m in prompt.messages],
            stream=True,
            **to_mistral_params(config),
        )

        finish_reason = None
        usage = None

        async for event in response:
            choice = event.data.choices[0]

            delta = choice.delta
            token = getattr(delta, "content", None)

            if isinstance(token, str):
                yield StreamChunk(delta=token)

            if choice.finish_reason:
                finish_reason = from_mistral_finish_reason(
                    choice.finish_reason
                )
            
            if event.data.usage:
                usage = TokenUsage(
                    input_tokens=event.data.usage.prompt_tokens or 0,
                    output_tokens=event.data.usage.completion_tokens or 0,
                )

        yield StreamChunk(
            delta="",
            finish_reason=finish_reason,
            usage=usage,
        )

            