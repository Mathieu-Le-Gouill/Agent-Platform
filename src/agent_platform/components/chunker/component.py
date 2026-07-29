from __future__ import annotations

from typing import Generic, TypeVar

from agent_platform.components.base import Component
from agent_platform.core.interfaces.chunking.base import BaseChunker
from agent_platform.core.interfaces.chunking.config import ChunkerConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.document import TextDocument

ChunkerConfigT = TypeVar("ChunkerConfigT", bound=ChunkerConfig)


class Chunker(
    Component[list[TextDocument], list[TextChunk]],
    Generic[ChunkerConfigT],
):
    def __init__(
        self,
        backend: BaseChunker[TextDocument, TextChunk, ChunkerConfigT],
        config: ChunkerConfigT | None = None,
    ) -> None:
        self._backend = backend
        self._config = config

    async def arun(self, input: list[TextDocument]) -> list[TextChunk]:
        return self._backend.chunk(input, self._config)
