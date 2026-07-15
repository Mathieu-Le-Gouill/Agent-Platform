from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from agent_platform.core.credentials import BaseCredentials
from agent_platform.core.interfaces.translation.config import TranslationConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.enums import Language

ConfigT = TypeVar("ConfigT", bound=TranslationConfig)
CredentialsT = TypeVar("CredentialsT", bound=BaseCredentials)


class BaseTranslator(ABC, Generic[CredentialsT, ConfigT]):
    def __init__(self, credentials: CredentialsT) -> None:
        self._credentials = credentials

    @abstractmethod
    async def translate(
        self, content: TextChunk, target: Language, source: Language | None = None
    ) -> TextChunk: ...
