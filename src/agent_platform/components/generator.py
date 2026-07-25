from __future__ import annotations

from typing import Generic, TypeVar

from agent_platform.components.base import Component
from agent_platform.core.interfaces.llm.base import BaseLLMProvider
from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.interfaces.llm.response import LLMResponse
from agent_platform.core.schemas.message import Prompt

GenerationConfigT = TypeVar("GenerationConfigT", bound=GenerationConfig)


class Generator(Component[Prompt, LLMResponse], Generic[GenerationConfigT]):
    def __init__(
        self,
        backend: BaseLLMProvider[GenerationConfigT],
        config: GenerationConfigT | None = None,
    ) -> None:
        self._backend = backend
        self._config = config

    async def arun(self, input: Prompt) -> LLMResponse:
        return await self._backend.agenerate(input, self._config)
