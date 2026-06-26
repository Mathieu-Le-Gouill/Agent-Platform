from typing import AsyncIterator

from anthropic import AsyncAnthropic

from platform.mappers.anthropic.generation_config import to_anthropic as to_anthropic_params
from mappers.anthropic.message import extract_system, to_anthropic as to_anthropic_message

from adapters.llm.prompt import Prompt
from adapters.llm.config import GenerationConfig
from adapters.llm.response import LLMResponse, StreamChunk
from mappers.anthropic.response import from_anthropic_response, from_anthropic_finish_reason
from platform.models.token import TokenUsage


class AnthropicLLM:
    client: AsyncAnthropic


    def __init__(self, api_key: str) -> None:
        self.client = AsyncAnthropic(api_key=api_key)


    async def generate(
        self,
        prompt: Prompt,
        model: str,
        config: GenerationConfig | None = None,
    ) -> LLMResponse:

        system, messages = extract_system(prompt)

        response = await self.client.messages.create(
            model=model,
            system=system or "",
            messages=[to_anthropic_message(m) for m in messages],
            **to_anthropic_params(config),
        )

        return from_anthropic_response(response)


    async def stream(
        self,
        prompt: Prompt,
        model: str,
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[StreamChunk]:

        system, messages = extract_system(prompt)

        finish_reason = None

        async with self.client.messages.stream(
            model=model,
            system=system or "",
            messages=[to_anthropic_message(m) for m in messages],
            **to_anthropic_params(config),
        ) as stream:

            async for text in stream.text_stream:
                yield StreamChunk(delta=text)

            final_message = await stream.get_final_message()

            if final_message.stop_reason:
                finish_reason = from_anthropic_finish_reason(final_message.stop_reason)

            yield StreamChunk(
                delta="",
                finish_reason=finish_reason,
                usage=TokenUsage(
                    input_tokens=final_message.usage.input_tokens,
                    output_tokens=final_message.usage.output_tokens,
                ),
            )