from abc import ABC, abstractmethod
from typing import AsyncIterator

from integrations.llm.prompt import Prompt
from integrations.llm.config import GenerationConfig
from integrations.llm.response import LLMResponse, StreamChunk


class BaseLLMProvider(ABC):

    @abstractmethod
    async def generate(
        self,
        prompt: Prompt,
        model: str,
        config: GenerationConfig | None = None,
    ) -> LLMResponse: ...

    @abstractmethod
    def stream(
        self,
        prompt: Prompt,
        model: str,
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[StreamChunk]: ...