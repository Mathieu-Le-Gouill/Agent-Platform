from __future__ import annotations

import asyncio
import threading
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import UUID

import faiss
import numpy as np

from agent_platform.core.errors import ValidationError
from agent_platform.core.interfaces.vector_store.base import BaseVectorStore
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import Score
from agent_platform.integrations.vector_store._base import reject_namespace
from agent_platform.integrations.vector_store.faiss.config import (
    DistanceMetric,
    FAISSConfig,
)
from agent_platform.integrations.vector_store.faiss.storage import (
    IndexState,
    chunk_to_json,
    read_state,
    write_state,
)

_STORE = "FAISS"
_MAX_IDS_SHOWN = 5


def _reject_repeated_ids(documents: list[TextChunk]) -> None:
    counts = Counter(doc.id for doc in documents)
    repeated = [str(chunk_id) for chunk_id, n in counts.items() if n > 1]
    if repeated:
        shown = ", ".join(repeated[:_MAX_IDS_SHOWN])
        more = len(repeated) - _MAX_IDS_SHOWN
        suffix = f" and {more} more" if more > 0 else ""
        raise ValidationError(
            f"FAISS add got repeated chunk ids in one batch: {shown}{suffix}"
        )


def _resolve_key(index_path: str | None) -> str | None:
    return str(Path(index_path).resolve()) if index_path else None


class FAISSStore(BaseVectorStore[FAISSConfig]):
    """Local, in-process vector store backed directly by `faiss-cpu`.

    Since this is offline/local compute with no network calls, it
    intentionally carries no `@error_logged`/`@with_retry` decorators.

    State is kept per resolved `index_path` (`None` = in memory only). Each
    operation runs as one worker-thread call holding that path's
    `threading.Lock`, so cancelling the awaiting coroutine never releases
    the lock while the thread still touches the index or its files. The
    lock is per instance: two instances or processes sharing one
    `index_path` are not coordinated.
    """

    def __init__(self) -> None:
        self._states: dict[str | None, IndexState] = {}
        self._locks: dict[str | None, threading.Lock] = {}
        self._locks_guard = threading.Lock()

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

    def _lock(self, key: str | None) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = self._locks[key] = threading.Lock()
            return lock

    def _state(self, key: str | None) -> IndexState | None:
        state = self._states.get(key)
        if state is None and key is not None:
            state = read_state(Path(key))
            if state is not None:
                self._states[key] = state
        return state

    def _persist(self, key: str | None, state: IndexState) -> None:
        if key is not None:
            write_state(Path(key), state)

    def _add_sync(
        self,
        documents: list[TextChunk],
        vectors: list[list[float]],
        config: FAISSConfig,
    ) -> None:
        key = _resolve_key(config.index_path)
        with self._lock(key):
            if key is not None:
                for doc in documents:
                    chunk_to_json(doc)
            try:
                state = self._state(key)
                if state is None:
                    dimension = len(vectors[0])
                    state = IndexState(
                        index=self._new_index(dimension, config.distance),
                        dimension=dimension,
                        distance=config.distance,
                    )
                    self._states[key] = state
                prepared = self._prepare(vectors, state.distance)

                ids: list[int] = []
                reused_ids: list[int] = []
                for doc in documents:
                    internal_id = state.uuid_to_id.get(doc.id)
                    if internal_id is None:
                        internal_id = state.next_id
                        state.next_id += 1
                        state.uuid_to_id[doc.id] = internal_id
                    else:
                        # Re-adding an existing chunk id: drop the stale vector
                        # first so this behaves as an upsert, not a duplicate.
                        reused_ids.append(internal_id)
                    state.docs[internal_id] = doc
                    ids.append(internal_id)

                if reused_ids:
                    state.index.remove_ids(np.asarray(reused_ids, dtype="int64"))  # type: ignore[arg-type]
                state.index.add_with_ids(prepared, np.asarray(ids, dtype="int64"))
                self._persist(key, state)
            except BaseException:
                self._forget(key)
                raise

    def _delete_sync(self, document_ids: list[UUID], config: FAISSConfig) -> None:
        key = _resolve_key(config.index_path)
        with self._lock(key):
            try:
                state = self._state(key)
                if state is None:
                    return
                internal_ids = [
                    state.uuid_to_id.pop(doc_id)
                    for doc_id in document_ids
                    if doc_id in state.uuid_to_id
                ]
                if not internal_ids:
                    return
                for internal_id in internal_ids:
                    state.docs.pop(internal_id, None)
                state.index.remove_ids(np.asarray(internal_ids, dtype="int64"))  # type: ignore[arg-type]
                self._persist(key, state)
            except BaseException:
                self._forget(key)
                raise

    def _forget(self, key: str | None) -> None:
        # A persisted index reloads its last saved files after a failure
        # instead of serving a half-applied change from memory.
        if key is not None:
            self._states.pop(key, None)

    def _search_sync(
        self,
        query_vector: list[float],
        k: int,
        config: FAISSConfig,
        filter: dict[str, Any] | None,
    ) -> list[tuple[TextChunk, Score]]:
        key = _resolve_key(config.index_path)
        with self._lock(key):
            state = self._state(key)
            if state is None or state.index.ntotal == 0:
                return []

            query = self._prepare([query_vector], state.distance)
            # FAISS has no server-side filter, so over-fetch when filtering
            # post-hoc to reduce the chance of ending up with fewer than k matches.
            fetch_k = k if not filter else min(state.index.ntotal, max(k * 4, k))
            distances, ids = state.index.search(query, fetch_k)

            results: list[tuple[TextChunk, Score]] = []
            for raw_distance, internal_id in zip(distances[0], ids[0], strict=True):
                if internal_id == -1:
                    continue
                chunk = state.docs.get(int(internal_id))
                if chunk is None or not self._matches(chunk, filter):
                    continue
                similarity = self._to_similarity(float(raw_distance), state.distance)
                results.append((chunk, Score.similarity(similarity)))
                if len(results) >= k:
                    break
            return results

    async def add(
        self,
        documents: list[TextChunk],
        vectors: list[list[float]],
        config: FAISSConfig | None = None,
    ) -> None:
        config = config or self._default_config()
        reject_namespace(config, _STORE)
        if not documents:
            return
        _reject_repeated_ids(documents)
        await asyncio.to_thread(self._add_sync, documents, vectors, config)

    async def delete(
        self, document_ids: list[UUID], config: FAISSConfig | None = None
    ) -> None:
        config = config or self._default_config()
        reject_namespace(config, _STORE)
        await asyncio.to_thread(self._delete_sync, document_ids, config)

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
        reject_namespace(config, _STORE)
        return await asyncio.to_thread(
            self._search_sync, query_vector, k, config, filter
        )

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
