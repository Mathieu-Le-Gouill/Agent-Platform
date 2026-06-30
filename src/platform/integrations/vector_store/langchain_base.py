from abc import abstractmethod
import asyncio

from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document as LC_Document

from integrations.vector_store.base import BaseVectorStore

from bridges.langchain.chunk import from_langchain as chunk_from_lc, from_langchain_many as chunks_from_lc, to_langchain as chunk_to_lc
from models.chunk import Chunk
from models.score import Score


class LangChainVectorStore(BaseVectorStore):

    _client: VectorStore


    def __init__(self):

        self._client = self._build_client()



    @abstractmethod
    def _build_client(
        self
    ) -> VectorStore:
        ...



    async def add(
        self,
        documents: list[Chunk],
    ) -> None:
        
        lc_docs: list[LC_Document] = [ 
            chunk_to_lc(document) 
            for document in documents
        ]

        await asyncio.to_thread(
            self._client.aadd_documents,
            lc_docs,
        )



    async def search(
        self,
        query_vector: list[float],
        k: int = 5,
    ) -> list[Chunk]:

        results: list[LC_Document] = await asyncio.to_thread(
            self._client.similarity_search_by_vector,
            query_vector,
            k,
        )

        return chunks_from_lc(results)
    

    async def search_with_scores(
        self,
        query_vector: list[float],
        k: int = 5,
    ) -> list[tuple[Chunk, Score]]:

        results = await asyncio.to_thread(
            self._client.similarity_search_with_score,
            query_vector,
            k,
        )

        return [
            (
                chunk_from_lc(doc),
                Score.similarity(
                    float(score)
                )
            )
            for doc, score in results
        ]