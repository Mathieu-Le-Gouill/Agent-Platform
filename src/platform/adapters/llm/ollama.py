from typing import AsyncIterator

from ollama import AsyncClient

from mappers.ollama.message import to_ollama as to_ollama_message
from mappers.ollama.generation_config import to_ollama as to_ollama_params

from adapters.llm.config import GenerationConfig
from adapters.llm.prompt import Prompt
from adapters.llm.response import LLMResponse, StreamChunk
from mappers.ollama.response import from_ollama_finish_reason, from_ollama_response
from models.token import TokenUsage


class OllamaLLM:
    client: AsyncClient


    def __init__(self, host: str) -> None:
        self.client = AsyncClient(host=host)


    async def generate(
        self,
        prompt: Prompt,
        model: str,
        config: GenerationConfig | None = None,
    ) -> LLMResponse:

        response = await self.client.chat(
            model=model,
            messages=[to_ollama_message(m) for m in prompt.messages],
            **to_ollama_params(config),
        )

        return from_ollama_response(response)
    

    async def stream(
        self,
        prompt: Prompt,
        model: str,
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[StreamChunk]:

        stream = await self.client.chat(
            model=model,
            messages=[to_ollama_message(m) for m in prompt.messages],
            stream=True,
            **to_ollama_params(config),
        )

        finish_reason = None
        usage = None

        async for chunk in stream:
            message = chunk.get("message", {})
            content = message.get("content")

            if content:
                yield StreamChunk(delta=content)
            
            if chunk.get("done"):
                if chunk.get("done_reason"):
                    finish_reason = from_ollama_finish_reason(
                        chunk.get("done_reason")
                    )

                usage = TokenUsage(
                    input_tokens=chunk.get("input_tokens", 0),
                    output_tokens=chunk.get("output_tokens", 0),
                )
                
        yield StreamChunk(
            delta="",
            finish_reason=finish_reason,
            usage=usage,
        )
