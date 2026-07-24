from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Generic, TypeVar

from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.interfaces.llm.response import LLMResponse, StreamChunk
from agent_platform.core.schemas.message import Prompt

if TYPE_CHECKING:
    from agent_platform.agents.tools.base import Tool

GenerationConfigT = TypeVar("GenerationConfigT", bound=GenerationConfig)


class BaseLLMProvider(ABC, Generic[GenerationConfigT]):
    @abstractmethod
    def generate(
        self,
        prompt: Prompt,
        config: GenerationConfigT | None = None,
        tools: list[Tool] | None = None,
    ) -> LLMResponse: ...

    @abstractmethod
    async def agenerate(
        self,
        prompt: Prompt,
        config: GenerationConfigT | None = None,
        tools: list[Tool] | None = None,
    ) -> LLMResponse: ...

    @abstractmethod
    def stream(
        self,
        prompt: Prompt,
        config: GenerationConfigT | None = None,
    ) -> AsyncIterator[StreamChunk]: ...
