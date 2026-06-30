from abc import abstractmethod
from typing import Sequence

from langchain_text_splitters import TextSplitter

from models.chunk import Chunk
from models.document import TextDocument

from integrations.chunking.base import BaseChunker
from integrations.chunking.config import ChunkerConfig

from bridges.langchain.document import to_langchain as doc_to_lc
from bridges.langchain.chunk import from_langchain as chunks_from_lc


class LangChainChunker(BaseChunker):


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
    ) -> list[Chunk]:

        splitter = self._splitter(config)

        for doc in documents:

        lc_documents = [doc_to_lc(doc) for doc in documents]

        lc_chunks = splitter.split_documents(lc_documents)

        return chunks_from_lc(lc_chunks)