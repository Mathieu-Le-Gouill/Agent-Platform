from abc import abstractmethod
from collections.abc import Sequence
from typing import Generic

from langchain_text_splitters import TextSplitter

from agent_platform.core.interfaces.chunking.base import (
    BaseChunker,
    ChunkerConfigT,
)
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.document import TextDocument
from agent_platform.integrations.chunking.mappers import doc_to_lc, lc_to_chunks


class LangChainChunker(
    BaseChunker[TextDocument, TextChunk, ChunkerConfigT],
    Generic[ChunkerConfigT],
):
    @abstractmethod
    def _splitter(self, config: ChunkerConfigT) -> TextSplitter: ...

    @abstractmethod
    def _default_config(self) -> ChunkerConfigT: ...

    def chunk(
        self,
        documents: Sequence[TextDocument],
        config: ChunkerConfigT | None,
    ) -> list[TextChunk]:
        config = config or self._default_config()

        splitter = self._splitter(config)
        lc_documents = [doc_to_lc(doc) for doc in documents]
        lc_chunks = splitter.split_documents(lc_documents)
        return lc_to_chunks(lc_chunks)
