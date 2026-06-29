from abc import abstractmethod
from typing import overload, Sequence

from langchain_core.documents import (
    BaseDocumentCompressor,
    Document as LC_Document,
)

from integrations.reranking.base import BaseReranker
from integrations.reranking.config import RerankerConfig

from models.chunk import Chunk
from models.document import Document
from models.protocols.text_unit import TextUnit

from bridges.langchain.chunk import (
    to_langchain as chunk_to_lc,
    from_langchain as chunk_from_lc,
)



class LangChainReranker(BaseReranker[TextUnit]):

    @abstractmethod
    def _client(self) -> BaseDocumentCompressor:
        ...


    @overload
    async def rerank(
        self,
        query: str,
        items: Sequence[Chunk],
        config: RerankerConfig | None = None,
    ) -> Sequence[Chunk]:
        ...


    @overload
    async def rerank(
        self,
        query: str,
        items: Sequence[Document],
        config: RerankerConfig | None = None,
    ) -> Sequence[Document]:
        ...


    async def rerank(
        self,
        query: str,
        items: Sequence[TextUnit],
        config: RerankerConfig | None = None,
    ) -> Sequence[TextUnit]:
        if not items:
            return []
 
        first = items[0]
 
        if isinstance(first, Chunk):
            return await self._rerank_chunks(query, items, config)  # type: ignore[arg-type]
 
        if isinstance(first, Document):
            return await self._rerank_documents(query, items, config)  # type: ignore[arg-type]
 
        raise NotImplementedError(
            f"{type(self).__name__} does not support TextUnit subtype "
            f"'{type(first).__name__}'. Add an isinstance branch and a "
            f"_rerank_{type(first).__name__.lower()}() method."
        )
    
    
    async def _rerank_documents(
        self,
        query: str,
        items: Sequence[Document],
        config: RerankerConfig | None,
    ) -> list[Document]:

        chunks = [
            chunk
            for doc in items
            for chunk in doc.chunks
        ]

        ranked_chunks = await self._rerank_chunks(
            query,
            chunks,
            None,  # top_k must be applied on documents
        )

        # Keep document order based on best-ranked chunk occurrence
        documents_by_id = {
            doc.id: doc
            for doc in items
        }

        result = []
        seen = set()

        for chunk in ranked_chunks:
            if chunk.document_id not in seen:
                result.append(
                    documents_by_id[chunk.document_id]
                )
                seen.add(chunk.document_id)

        if config is not None and config.top_k is not None:
            result = result[:config.top_k]

        return result
        

    async def _rerank_chunks( 
        self,
        query: str,
        items: Sequence[Chunk],
        config: RerankerConfig | None = None,
    ) -> list[Chunk]:
        client = self._client()
        
        documents: list[LC_Document] = [chunk_to_lc(item) for item in items]
        
        result = await client.acompress_documents(documents, query)
        
        if config is not None and config.top_k is not None:
            result = result[: config.top_k]
            
        return [chunk_from_lc(item) for item in result]