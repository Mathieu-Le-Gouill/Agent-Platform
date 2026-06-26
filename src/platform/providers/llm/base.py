from typing import Protocol
from typing import AsyncIterator

from providers.llm.response import LLMResponse, StreamChunk
from providers.llm.prompt import Prompt
from providers.llm.config import GenerationConfig


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