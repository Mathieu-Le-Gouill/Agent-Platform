# adapters/vector_store/chroma_store.py
from __future__ import annotations

from uuid import UUID

import chromadb
from chromadb.api import AsyncClientAPI
from chromadb.api.models.AsyncCollection import AsyncCollection

from models.chunk import Chunk
from models.document import Document
from models.score import Score
from agent_platform.bridges.chroma.chunk import to_chroma, from_chroma


class ChromaStore: # implements VectorStorePort

    def __init__(self, host: str, port: int, collection_name: str) -> None:
        self._host = host
        self._port = port
        self._collection_name = collection_name
        self._client: AsyncClientAPI | None = None
        self._collection: AsyncCollection | None = None


    async def _ensure_client(self) -> AsyncClientAPI:

        if self._client is None:
            self._client = await chromadb.AsyncHttpClient(
                host=self._host, 
                port=self._port
            )

        return self._client
    

    async def _get_collection(self) -> AsyncCollection:

        if self._collection is None:
            client = await self._ensure_client()

            self._collection = await client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        return self._collection


    async def add(self, documents: list[Document]) -> None:

        collection = await self._get_collection()
        ids, embeddings, metadatas, texts = [], [], [], []

        for document in documents:
            for chunk in document.chunks:
                row = to_chroma(chunk)
                ids.append(row["id"])
                embeddings.append(row["embedding"])
                metadatas.append(row["metadata"])
                texts.append(row["text"])

        await collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts,
        )


    async def delete(self, document_ids: list[UUID]) -> None:

        collection = await self._get_collection()
        
        await collection.delete(
            where={"document_id": {"$in": [str(did) for did in document_ids]}}
        )


    async def search(self, query_vector: list[float], k: int = 5) -> list[Chunk]:
        
        chunks, _ = await self._query(query_vector, k)
        return chunks


    async def search_with_scores(
        self, query_vector: list[float], k: int = 5
    ) -> list[tuple[Chunk, Score]]:
        
        chunks, distances = await self._query(query_vector, k)
        scores = [
            Score.similarity(max(-1.0, min(1.0, 1.0 - d)))
            for d in distances
        ]
        return list(zip(chunks, scores))


    async def _query(
        self, query_vector: list[float], k: int
    ) -> tuple[list[Chunk], list[float]]:
        
        collection = await self._get_collection()
        response = await collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        raw_docs      = response.get("documents")      or [[]]
        raw_metadatas = response.get("metadatas")      or [[]]
        raw_distances = response.get("distances")      or [[]]

        docs      = raw_docs[0]      if raw_docs      else []
        metadatas = raw_metadatas[0] if raw_metadatas else []
        distances = raw_distances[0] if raw_distances else []

        chunks = [
            from_chroma(doc, dict(meta)) 
            for doc, meta in zip(docs, metadatas)
        ]
        return chunks, distances