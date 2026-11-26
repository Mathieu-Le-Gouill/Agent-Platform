from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Generic, TypeVar

from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.interfaces.llm.response import LLMResponse, StreamChunk
from agent_platform.core.interfaces.llm.tool_spec import ToolSpec
from agent_platform.core.schemas.message import Prompt

GenerationConfigT = TypeVar("GenerationConfigT", bound=GenerationConfig)


class BaseLLMProvider(ABC, Generic[GenerationConfigT]):
    @abstractmethod
    def generate(
        self,
        prompt: Prompt,
        config: GenerationConfigT | None = None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> LLMResponse: ...

    @abstractmethod
    async def agenerate(
        self,
        prompt: Prompt,
        config: GenerationConfigT | None = None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> LLMResponse:
        """Generate a response; failures raise typed errors (see `integrations/README.md`)."""

    @abstractmethod
    def stream(
        self,
        prompt: Prompt,
        config: GenerationConfigT | None = None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a response; failures raise typed errors (see `integrations/README.md`)."""
