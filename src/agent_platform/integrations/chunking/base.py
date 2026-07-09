from abc import ABC, abstractmethod

from typing import Generic, Sequence, TypeVar

from agent_platform.models.document import Document
from agent_platform.models.chunk import Chunk

from agent_platform.integrations.credentials import BaseCredentials
from agent_platform.integrations.chunking.config import ChunkerConfig

Document_T = TypeVar("Document_T", bound=Document, contravariant=True)
Chunk_T = TypeVar("Chunk_T", bound=Chunk)
ChunkerConfigT = TypeVar("ChunkerConfigT", bound=ChunkerConfig)
CredentialsT = TypeVar("CredentialsT", bound=BaseCredentials)

class BaseChunker(ABC, Generic[CredentialsT, Document_T, Chunk_T, ChunkerConfigT]):

    def __init__(self, credentials: CredentialsT) -> None:
        self._credentials = credentials

    @abstractmethod
    def chunk(
        self,
        documents: Sequence[Document_T],
        config: ChunkerConfigT | None,
    ) -> list[Chunk_T]: ...