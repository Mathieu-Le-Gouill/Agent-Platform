from __future__ import annotations
from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, Filter, FieldCondition, MatchAny, VectorParams

from models.chunk import Chunk
from models.document import Document
from models.score import Score
from bridges.chunk.qdrant import to_point, from_point


class QdrantStore:

    def __init__(
        self,
        url: str,
        api_key: str,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
    ) -> None:
        self._client = AsyncQdrantClient(url=url, api_key=api_key)
        self._collection = collection_name
        self._vector_size = vector_size
        self._distance = distance


    @classmethod
    async def create(
        cls,
        url: str,
        api_key: str,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE
    ) -> QdrantStore:
        
        instance = cls(url, api_key, collection_name, vector_size, distance)
        await instance.ensure_collection()
        return instance


    async def ensure_collection(self) -> None:
        exists = await self._client.collection_exists(self._collection)
        if not exists:
            await self._client.create_collection(
                self._collection,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=self._distance,
                ),
            )


    async def add(self, documents: list[Document]) -> None:
        points = [
            to_point(chunk)
            for document in documents
            for chunk in document.chunks
        ]
        await self._client.upsert(collection_name=self._collection, points=points)


    async def delete(self, document_ids: list[UUID]) -> None:
        await self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchAny(any=[str(did) for did in document_ids]),
                    )
                ]
            ),
        )


    async def search(self, query_vector: list[float], k: int = 5) -> list[Chunk]:
        results = await self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=k,
            with_payload=True,
        )
        return [from_point(r) for r in results.points]


    async def search_with_scores(
        self, query_vector: list[float], k: int = 5
    ) -> list[tuple[Chunk, Score]]:
        results = await self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=k,
            with_payload=True,
        )
        return [
            (from_point(r), Score.similarity(max(-1.0, min(1.0, r.score))))
            for r in results.points
        ]