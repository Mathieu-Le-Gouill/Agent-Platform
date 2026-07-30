from __future__ import annotations

from typing import Generic, TypeVar

from agent_platform.components.base import Component
from agent_platform.core.interfaces.chunking.base import BaseChunker
from agent_platform.core.interfaces.chunking.config import ChunkerConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.document import TextDocument

ChunkerConfigT = TypeVar("ChunkerConfigT", bound=ChunkerConfig)

ChunkerInput = tuple[list[TextDocument], ChunkerConfigT | None]


class Chunker(
    Component[ChunkerInput[ChunkerConfigT], list[TextChunk]],
    Generic[ChunkerConfigT],
):
    def __init__(
        self,
        backend: BaseChunker[TextDocument, TextChunk, ChunkerConfigT],
    ) -> None:
        self._backend = backend

    async def arun(self, input: ChunkerInput[ChunkerConfigT]) -> list[TextChunk]:
        documents, config = input
        return self._backend.chunk(documents, config)
