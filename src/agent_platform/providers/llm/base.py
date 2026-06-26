from typing import Protocol
from typing import AsyncIterator

from models.message import AssistantMessage
from models.prompt import Prompt
from models.generation import GenerationConfig


class LLMProvider(Protocol):
    async def generate(
        self,
        prompt: Prompt,
        model: str,
        config: GenerationConfig | None = None,
    ) -> AssistantMessage | None:
        ...


    async def stream(
        self,
        prompt: Prompt,
        model: str,
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[str]:
        ...


    async def chat(
        self,
        prompt: Prompt,
        model: str,
        config: GenerationConfig | None = None,
    ) -> AssistantMessage | None: 
        ...