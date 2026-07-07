from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, AsyncIterator, Generic, TypeVar

from agent_platform.models.message import Prompt
from agent_platform.integrations.llm.config import GenerationConfig
from agent_platform.integrations.llm.response import LLMResponse, StreamChunk

if TYPE_CHECKING:
    from agent_platform.agents.tools.base import Tool

GenerationConfigT = TypeVar("GenerationConfigT", bound=GenerationConfig)


class BaseLLMProvider(ABC, Generic[GenerationConfigT]):
    @abstractmethod
    async def generate(
        self,
        prompt: Prompt,
        model: str,
        config: GenerationConfigT | None = None,
        tools: list[Tool] | None = None,
    ) -> LLMResponse: ...

    @abstractmethod
    def stream(
        self,
        prompt: Prompt,
        model: str,
        config: GenerationConfigT | None = None,
    ) -> AsyncIterator[StreamChunk]: ...
