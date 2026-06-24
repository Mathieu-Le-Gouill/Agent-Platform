from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, Filter, FieldCondition, MatchAny, VectorParams

from core.entities.chunk import Chunk
from core.entities.document import Document
from core.value_objects.score import Score
from adapters.vector_store._mapper import chunk_to_point, point_to_chunk


class QdrantStore:  # implements VectorStorePort

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
            chunk_to_point(chunk, document)
            for document in documents
            for chunk in document.chunks
        ]
        await self._client.upsert(collection_name=self._collection, points=points)


    async def delete(self, document_ids: list[UUID]) -> None:
        await self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                should=[
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
        return [point_to_chunk(r) for r in results.points]


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
            (point_to_chunk(r), Score.similarity(max(-1.0, min(1.0, r.score))))
            for r in results.points
        ]