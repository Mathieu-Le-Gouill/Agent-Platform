from typing import Protocol
from typing import AsyncIterator

from adapters.llm.response import LLMResponse, StreamChunk
from adapters.llm.prompt import Prompt
from adapters.llm.config import GenerationConfig


class LLMProvider(Protocol):
    async def generate(
        self,
        prompt: Prompt,
        model: str,
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        ...


    async def stream(
        self,
        prompt: Prompt,
        model: str,
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[StreamChunk]:
        ...