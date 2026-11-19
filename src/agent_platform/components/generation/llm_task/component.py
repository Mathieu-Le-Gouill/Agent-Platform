from __future__ import annotations

from typing import Generic, NamedTuple, TypeVar

from agent_platform.components.base import Component
from agent_platform.core.interfaces.llm.base import BaseLLMProvider
from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.interfaces.llm.structured import (
    StructuredResult,
    generate_structured,
)
from agent_platform.core.schemas.message import Prompt

T = TypeVar("T")


class LLMTaskInput(NamedTuple):
    text: str
    config: GenerationConfig | None


class LLMTask(Component[LLMTaskInput, StructuredResult[T]], Generic[T]):
    """Runs one instruction over a text and returns the answer as `output_type`.

    The instruction is the system prompt and the text is the user message. The
    model comes from `config.model`, else from the model the provider is bound to.
    """

    def __init__(
        self, llm: BaseLLMProvider, instructions: str, output_type: type[T]
    ) -> None:
        self._llm = llm
        self._instructions = instructions
        self._output_type = output_type

    async def arun(self, input: LLMTaskInput) -> StructuredResult[T]:
        text, config = input
        prompt = Prompt.build(system=self._instructions, user=text)
        return await generate_structured(self._llm, prompt, self._output_type, config)
