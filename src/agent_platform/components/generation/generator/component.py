from __future__ import annotations

from typing import Generic, NamedTuple, TypeVar

from agent_platform.components.base import Component
from agent_platform.core.interfaces.llm.base import BaseLLMProvider
from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.interfaces.llm.response import LLMResponse
from agent_platform.core.schemas.message import Prompt

GenerationConfigT = TypeVar("GenerationConfigT", bound=GenerationConfig)


class GeneratorInput(NamedTuple, Generic[GenerationConfigT]):
    prompt: Prompt
    config: GenerationConfigT | None


class Generator(
    Component[GeneratorInput[GenerationConfigT], LLMResponse],
    Generic[GenerationConfigT],
):
    def __init__(
        self,
        backend: BaseLLMProvider[GenerationConfigT],
    ) -> None:
        self._backend = backend

    async def arun(self, input: GeneratorInput[GenerationConfigT]) -> LLMResponse:
        prompt, config = input
        return await self._backend.agenerate(prompt, config)
