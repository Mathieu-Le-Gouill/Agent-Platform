from abc import ABC, abstractmethod
from typing import AsyncIterator, Generic, TypeVar

from agent_platform.integrations.llm.prompt import Prompt
from agent_platform.integrations.llm.config import GenerationConfig
from agent_platform.integrations.llm.response import LLMResponse, StreamChunk

GenerationConfigT = TypeVar("GenerationConfigT", bound=GenerationConfig)


class BaseLLMProvider(ABC, Generic[GenerationConfigT]):

    @abstractmethod
    async def generate(
        self,
        prompt: Prompt,
        model: str,
        config: GenerationConfigT | None = None,
    ) -> LLMResponse: ...

    @abstractmethod
    def stream(
        self,
        prompt: Prompt,
        model: str,
        config: GenerationConfigT | None = None,
    ) -> AsyncIterator[StreamChunk]: ...
