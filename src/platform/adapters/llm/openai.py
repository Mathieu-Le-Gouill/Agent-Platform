from typing import AsyncIterator

from openai import AsyncOpenAI

from platform.mappers.openai.generation_config import to_openai as to_openai_params
from mappers.openai.message import to_openai as to_openai_message

from adapters.llm.config import GenerationConfig
from adapters.llm.prompt import Prompt
from adapters.llm.response import LLMResponse, StreamChunk
from mappers.openai.response import from_openai_finish_reason, from_openai_response
from models.token import TokenUsage


class OpenAILLM:
    client: AsyncOpenAI


    def __init__(self, api_key: str) -> None:
        self.client = AsyncOpenAI(api_key=api_key)

    async def generate(
        self,
        prompt: Prompt,
        model: str,
        config: GenerationConfig | None = None,
    ) -> LLMResponse:

        response = await self.client.chat.completions.create(
            model=model,
            messages=[to_openai_message(m) for m in prompt.messages],
            **to_openai_params(config),
        )

        return from_openai_response(response)

    

    async def stream(
        self,
        prompt: Prompt,
        model: str,
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[StreamChunk]:

        stream = await self.client.chat.completions.create(
            model=model,
            messages=[to_openai_message(m) for m in prompt.messages],
            **to_openai_params(config),
            stream_options={"include_usage": True},
            stream=True,
        )

        finish_reason = None
        usage = None

        async for chunk in stream:
            if not chunk.choices:
                if chunk.usage:
                    usage = TokenUsage(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                    )
                continue

            choice = chunk.choices[0]

            delta = choice.delta

            if delta.content:
                yield StreamChunk(delta=delta.content)

            if choice.finish_reason:
                finish_reason = from_openai_finish_reason(
                    choice.finish_reason
                )

        yield StreamChunk(
            delta="",
            finish_reason=finish_reason,
            usage=usage,
        )