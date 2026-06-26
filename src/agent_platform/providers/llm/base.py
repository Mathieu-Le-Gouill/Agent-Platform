from typing import Protocol
from typing import AsyncIterator

from models.message import AssistantMessage
from models.prompt import Prompt


class LLMProvider(Protocol):
    async def generate(
        self,
        prompt: Prompt,
        model: str,
    ) -> AssistantMessage | None:
        ...


    async def stream(
        self,
        prompt: Prompt,
        model: str,
    ) -> AsyncIterator[str]:
        ...


    async def chat(
        self,
        prompt: Prompt,
        model: str,
    ) -> AssistantMessage | None: 
        ...