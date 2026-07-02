from abc import abstractmethod
from typing import Sequence

from langchain_text_splitters import TextSplitter

from agent_platform.models.chunk import TextChunk
from agent_platform.models.document import TextDocument

from agent_platform.integrations.chunking.base import BaseChunker
from agent_platform.integrations.chunking.config import ChunkerConfig

from agent_platform.bridges.langchain.document import to_langchain as doc_to_lc
from agent_platform.bridges.langchain.chunk import from_langchain_many as chunks_from_lc


class LangChainChunker(BaseChunker[TextDocument, TextChunk]):


    @abstractmethod
    def _splitter(
        self,
        config: ChunkerConfig | None,
    ) -> TextSplitter:
        ...


    async def chunk(
        self,
        documents: Sequence[TextDocument],
        config: ChunkerConfig | None = None,
    ) -> list[TextChunk]:

        splitter = self._splitter(config)

        lc_documents = [doc_to_lc(doc) for doc in documents]

        lc_chunks = splitter.split_documents(lc_documents)

        return chunks_from_lc(lc_chunks)