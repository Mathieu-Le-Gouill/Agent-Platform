from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, AsyncIterator, Generic, TypeVar

from agent_platform.core.schemas.message import Prompt
from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.credentials import BaseCredentials
from agent_platform.core.interfaces.llm.response import LLMResponse, StreamChunk

if TYPE_CHECKING:
    from agent_platform.agents.tools.base import Tool

CredentialsT = TypeVar("CredentialsT", bound=BaseCredentials)
GenerationConfigT = TypeVar("GenerationConfigT", bound=GenerationConfig)


class BaseLLMProvider(ABC, Generic[CredentialsT, GenerationConfigT]):
    def __init__(self, credentials: CredentialsT) -> None:
        self._credentials = credentials

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
