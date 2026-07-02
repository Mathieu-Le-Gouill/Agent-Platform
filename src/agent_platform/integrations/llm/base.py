from abc import ABC, abstractmethod
from typing import AsyncIterator

from agent_platform.integrations.llm.prompt import Prompt
from agent_platform.integrations.llm.config import GenerationConfig
from agent_platform.integrations.llm.response import LLMResponse, StreamChunk


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