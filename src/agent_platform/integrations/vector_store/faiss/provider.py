from __future__ import annotations

import asyncio
import pickle
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import faiss
import numpy as np

from agent_platform.core.interfaces.vector_store.base import BaseVectorStore
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import Score
from agent_platform.integrations.vector_store.faiss.config import (
    DistanceMetric,
    FAISSConfig,
)

_INDEX_FILENAME = "index.faiss"
_DOCSTORE_FILENAME = "docstore.pkl"


class FAISSStore(BaseVectorStore[FAISSConfig]):
    """Local, in-process vector store backed directly by `faiss-cpu`.

    Since this is offline/local compute with no network calls, it
    intentionally carries no `@error_logged`/`@with_retry` decorators.
    """

    def __init__(self) -> None:
        self._index: faiss.IndexIDMap2 | None = None
        self._docs: dict[int, TextChunk] = {}
        self._uuid_to_id: dict[UUID, int] = {}
        self._next_id = 0
        self._dimension: int | None = None
        self._distance: DistanceMetric | None = None

    def _default_config(self) -> FAISSConfig:
        return FAISSConfig()

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def _prepare(
        self, vectors: list[list[float]], distance: DistanceMetric
    ) -> np.ndarray:
        array = np.asarray(vectors, dtype="float32")
        if distance is DistanceMetric.COSINE:
            array = self._normalize(array)
        return array

    def _new_index(self, dimension: int, distance: DistanceMetric) -> faiss.IndexIDMap2:
        # COSINE searches over pre-normalized vectors, so plain inner product
        # is equivalent to cosine similarity; DOT uses inner product directly
        # on the raw (un-normalized) vectors.
        base = (
            faiss.IndexFlatL2(dimension)
            if distance is DistanceMetric.EUCLIDEAN
            else faiss.IndexFlatIP(dimension)
        )
        return faiss.IndexIDMap2(base)

    def _load_or_none(self, config: FAISSConfig) -> faiss.IndexIDMap2 | None:
        if self._index is not None:
            return self._index
        if not config.index_path:
            return None
        path = Path(config.index_path)
        index_file = path / _INDEX_FILENAME
        if not index_file.exists():
            return None

        self._index = cast(faiss.IndexIDMap2, faiss.read_index(str(index_file)))
        with open(path / _DOCSTORE_FILENAME, "rb") as f:
            state = pickle.load(f)
        self._docs = state["docs"]
        self._uuid_to_id = state["uuid_to_id"]
        self._next_id = state["next_id"]
        self._dimension = state["dimension"]
        self._distance = state["distance"]
        return self._index

    async def _save(self, config: FAISSConfig) -> None:
        if not config.index_path or self._index is None:
            return
        path = Path(config.index_path)
        path.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(
            faiss.write_index, self._index, str(path / _INDEX_FILENAME)
        )
        state = {
            "docs": self._docs,
            "uuid_to_id": self._uuid_to_id,
            "next_id": self._next_id,
            "dimension": self._dimension,
            "distance": self._distance,
        }
        await asyncio.to_thread(self._write_pickle, path / _DOCSTORE_FILENAME, state)

    @staticmethod
    def _write_pickle(path: Path, state: dict[str, Any]) -> None:
        with open(path, "wb") as f:
            pickle.dump(state, f)

    async def add(
        self,
        documents: list[TextChunk],
        vectors: list[list[float]],
        config: FAISSConfig | None = None,
    ) -> None:
        config = config or self._default_config()
        if not documents:
            return

        if self._index is None:
            self._load_or_none(config)
        if self._index is None:
            self._dimension = len(vectors[0])
            self._distance = config.distance
            self._index = self._new_index(self._dimension, self._distance)

        distance = self._distance or config.distance
        prepared = self._prepare(vectors, distance)

        ids: list[int] = []
        reused_ids: list[int] = []
        for doc in documents:
            internal_id = self._uuid_to_id.get(doc.id)
            if internal_id is None:
                internal_id = self._next_id
                self._next_id += 1
                self._uuid_to_id[doc.id] = internal_id
            else:
                # Re-adding an existing chunk id: drop the stale vector first
                # so this behaves as an upsert rather than a duplicate insert.
                reused_ids.append(internal_id)
            self._docs[internal_id] = doc
            ids.append(internal_id)

        if reused_ids:
            await asyncio.to_thread(
                self._index.remove_ids,
                np.asarray(reused_ids, dtype="int64"),  # type: ignore[arg-type]
            )
        await asyncio.to_thread(
            self._index.add_with_ids, prepared, np.asarray(ids, dtype="int64")
        )

        await self._save(config)

    async def delete(
        self, document_ids: list[UUID], config: FAISSConfig | None = None
    ) -> None:
        config = config or self._default_config()
        index = self._load_or_none(config)
        if index is None:
            return

        internal_ids = [
            self._uuid_to_id.pop(doc_id)
            for doc_id in document_ids
            if doc_id in self._uuid_to_id
        ]
        if not internal_ids:
            return
        for internal_id in internal_ids:
            self._docs.pop(internal_id, None)

        await asyncio.to_thread(
            index.remove_ids,
            np.asarray(internal_ids, dtype="int64"),  # type: ignore[arg-type]
        )
        await self._save(config)

    def _matches(self, chunk: TextChunk, filter: dict[str, Any] | None) -> bool:
        if not filter:
            return True
        metadata = chunk.metadata or {}
        return all(metadata.get(key) == value for key, value in filter.items())

    def _to_similarity(self, raw: float, distance: DistanceMetric) -> float:
        if distance is DistanceMetric.EUCLIDEAN:
            similarity = 1.0 / (1.0 + raw)
        else:
            # COSINE (normalized inner product) and DOT (raw inner product)
            # are already similarity-shaped.
            similarity = raw
        return min(1.0, max(0.0, similarity))

    async def _search(
        self,
        query_vector: list[float],
        k: int,
        config: FAISSConfig | None,
        filter: dict[str, Any] | None,
    ) -> list[tuple[TextChunk, Score]]:
        config = config or self._default_config()
        index = self._load_or_none(config)
        if index is None or index.ntotal == 0:
            return []

        distance = self._distance or config.distance
        query = self._prepare([query_vector], distance)
        # FAISS has no server-side filter, so over-fetch when filtering post-hoc
        # to reduce the chance of ending up with fewer than k matches.
        fetch_k = k if not filter else min(index.ntotal, max(k * 4, k))
        distances, ids = await asyncio.to_thread(index.search, query, fetch_k)

        results: list[tuple[TextChunk, Score]] = []
        for raw_distance, internal_id in zip(distances[0], ids[0]):
            if internal_id == -1:
                continue
            chunk = self._docs.get(int(internal_id))
            if chunk is None or not self._matches(chunk, filter):
                continue
            results.append(
                (
                    chunk,
                    Score.similarity(
                        self._to_similarity(float(raw_distance), distance)
                    ),
                )
            )
            if len(results) >= k:
                break
        return results

    async def search(
        self,
        query_vector: list[float],
        k: int = 5,
        config: FAISSConfig | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        results = await self._search(query_vector, k, config, filter)
        return [chunk for chunk, _ in results]

    async def search_with_scores(
        self,
        query_vector: list[float],
        k: int = 5,
        config: FAISSConfig | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[TextChunk, Score]]:
        return await self._search(query_vector, k, config, filter)
