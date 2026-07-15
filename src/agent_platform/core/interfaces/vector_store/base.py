from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Generic, TypeVar
from uuid import UUID

from agent_platform.core.credentials import BaseCredentials
from agent_platform.core.interfaces.vector_store.config import VectorStoreConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import Score

ConfigT = TypeVar("ConfigT", bound=VectorStoreConfig)
CredentialsT = TypeVar("CredentialsT", bound=BaseCredentials)


class BaseVectorStore(ABC, Generic[CredentialsT, ConfigT]):
    def __init__(self, credentials: CredentialsT) -> None:
        self._credentials = credentials

    @abstractmethod
    async def add(
        self, documents: list[TextChunk], config: ConfigT | None = None
    ) -> None: ...

    @abstractmethod
    async def delete(
        self, document_ids: list[UUID], config: ConfigT | None = None
    ) -> None: ...

    @abstractmethod
    async def search(
        self, query_vector: list[float], k: int = 5, config: ConfigT | None = None
    ) -> list[TextChunk]: ...

    @abstractmethod
    async def search_with_scores(
        self, query_vector: list[float], k: int = 5, config: ConfigT | None = None
    ) -> list[tuple[TextChunk, Score]]: ...
