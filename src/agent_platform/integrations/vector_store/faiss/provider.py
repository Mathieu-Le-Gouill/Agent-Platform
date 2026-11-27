from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import faiss
import numpy as np
from pydantic_core import PydanticSerializationError

from agent_platform.core.errors import ConfigError, ValidationError
from agent_platform.core.interfaces.vector_store.base import BaseVectorStore
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import Score
from agent_platform.integrations.vector_store._base import reject_namespace
from agent_platform.integrations.vector_store.faiss.config import (
    DistanceMetric,
    FAISSConfig,
)

_STORE = "FAISS"
_INDEX_FILE_PATTERN = re.compile(r"index-[0-9a-f]{64}\.faiss")
_TEMP_FILE_PATTERN = re.compile(
    r"\.(docstore\.json|index-[0-9a-f]{64}\.faiss)\.[0-9a-f]{32}\.tmp"
)
logger = logging.getLogger(__name__)
_DOCSTORE_FILENAME = "docstore.json"
_LEGACY_DOCSTORE_FILENAME = "docstore.pkl"
_DOCSTORE_VERSION = 1
_REINGEST = "delete the directory and re-ingest its documents"
_MAX_IDS_SHOWN = 5

# Written as-is: the serialized index is a uint8 array, never copied to bytes.
Buffer = bytes | np.ndarray


@dataclass
class _IndexState:
    index: faiss.IndexIDMap2
    dimension: int
    distance: DistanceMetric
    docs: dict[int, TextChunk] = field(default_factory=dict)
    uuid_to_id: dict[UUID, int] = field(default_factory=dict)
    next_id: int = 0
    # name of the index file the committed docstore.json references, if any
    index_file: str | None = None


def _chunk_to_json(chunk: TextChunk) -> dict[str, Any]:
    try:
        return chunk.model_dump(mode="json")
    except PydanticSerializationError as exc:
        raise ValidationError(
            f"FAISS stores chunk metadata as JSON; chunk {chunk.id} has a value "
            f"JSON cannot hold: {exc}"
        ) from exc


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


def _index_filename(sha256: str) -> str:
    return f"index-{sha256}.faiss"


def _read_state(directory: Path) -> _IndexState | None:
    docstore_file = directory / _DOCSTORE_FILENAME
    if not docstore_file.exists():
        if (directory / _LEGACY_DOCSTORE_FILENAME).exists():
            raise ConfigError(
                f"FAISS index at {directory} has a legacy pickle docstore "
                f"({_LEGACY_DOCSTORE_FILENAME}), which is no longer loaded because "
                f"unpickling can run arbitrary code; {_REINGEST}"
            )
        # No commit yet: an index file here is an orphan of an interrupted
        # first save, removed by the next save.
        return None

    try:
        payload = json.loads(docstore_file.read_text(encoding="utf-8"))
        if payload["version"] != _DOCSTORE_VERSION:
            raise ValueError(f"unsupported version {payload['version']!r}")
        docs = {
            int(entry["id"]): TextChunk.model_validate(entry["chunk"])
            for entry in payload["docs"]
        }
        dimension = int(payload["dimension"])
        distance = DistanceMetric(payload["distance"])
        next_id = int(payload["next_id"])
        expected_sha256 = str(payload["index_sha256"])
        expected_ntotal = int(payload["ntotal"])
        index_name = str(payload["index_file"])
        # the name comes from the file, so it must not be able to point
        # anywhere but this directory's own content-addressed index
        if index_name != _index_filename(expected_sha256) or not (
            _INDEX_FILE_PATTERN.fullmatch(index_name)
        ):
            raise ValueError(f"unexpected index_file {index_name!r}")
    except (ValueError, KeyError, TypeError) as exc:
        raise ConfigError(
            f"FAISS docstore at {docstore_file} is invalid: {exc}"
        ) from exc

    index_file = directory / index_name
    if not index_file.exists():
        raise ConfigError(
            f"FAISS docstore at {directory} references a missing {index_name}; "
            f"{_REINGEST}"
        )
    # Hash and parse the same bytes, so the file cannot change in between.
    index_bytes = index_file.read_bytes()
    if hashlib.sha256(index_bytes).hexdigest() != expected_sha256:
        raise ConfigError(
            f"FAISS index at {directory} does not match its docstore (an "
            f"interrupted save?); {_REINGEST}"
        )
    try:
        index = cast(
            faiss.IndexIDMap2,
            faiss.deserialize_index(np.frombuffer(index_bytes, dtype="uint8")),
        )
    except Exception as exc:
        raise ConfigError(
            f"FAISS index at {index_file} cannot be read: {exc}; {_REINGEST}"
        ) from exc
    if not (
        index.ntotal == len(docs) == expected_ntotal
        and all(internal_id < next_id for internal_id in docs)
    ):
        raise ConfigError(
            f"FAISS index and docstore at {directory} are inconsistent; {_REINGEST}"
        )

    return _IndexState(
        index=index,
        dimension=dimension,
        distance=distance,
        docs=docs,
        uuid_to_id={chunk.id: internal_id for internal_id, chunk in docs.items()},
        next_id=next_id,
        index_file=index_name,
    )


def _write_temp(directory: Path, name: str, data: Buffer) -> Path:
    # os.open with 0o666 lets the umask set the final mode, as a plain
    # open() would; mkstemp would force 0o600.
    temp = directory / f".{name}.{uuid.uuid4().hex}.tmp"
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return temp


def _fsync_directory(directory: Path) -> None:
    flags = getattr(os, "O_DIRECTORY", None)
    if flags is None:  # not POSIX (Windows): directory fsync is unsupported
        return
    fd = os.open(directory, os.O_RDONLY | flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _remove_stray_files(directory: Path, keep: str | None) -> None:
    # Leftovers of interrupted saves only: temp files named as `_write_temp`
    # names them, and index files the committed docstore.json does not
    # reference. Anything else in the directory is left alone.
    for path in directory.iterdir():
        name = path.name
        if _TEMP_FILE_PATTERN.fullmatch(name) or (
            name != keep and _INDEX_FILE_PATTERN.fullmatch(name)
        ):
            path.unlink(missing_ok=True)


def _replace_from_temp(directory: Path, name: str, data: Buffer) -> None:
    temp = _write_temp(directory, name, data)
    try:
        os.replace(temp, directory / name)
    finally:
        temp.unlink(missing_ok=True)


def _after_commit(directory: Path, keep: str) -> None:
    # The change is already on disk: a failure here must not fail the call.
    try:
        _fsync_directory(directory)
        _remove_stray_files(directory, keep=keep)
    except OSError as exc:
        logger.warning(
            "FAISS cleanup after commit failed on %s: %s",
            Path(exc.filename).name if exc.filename else "<directory>",
            type(exc).__name__,
        )


def _write_state(directory: Path, state: _IndexState) -> None:
    # Replacing docstore.json is the only commit step. The index goes first
    # under a new content-addressed name, so the committed docstore keeps
    # pointing at its own, untouched index until the commit; a failure at
    # any point leaves the previous pair loadable plus at most one orphan.
    directory.mkdir(parents=True, exist_ok=True)
    _remove_stray_files(directory, keep=state.index_file)
    serialized = faiss.serialize_index(state.index)
    sha256 = hashlib.sha256(serialized).hexdigest()
    index_name = _index_filename(sha256)
    if not (directory / index_name).exists():
        _replace_from_temp(directory, index_name, serialized)
        _fsync_directory(directory)
    docstore = {
        "version": _DOCSTORE_VERSION,
        "dimension": state.dimension,
        "distance": state.distance.value,
        "next_id": state.next_id,
        "ntotal": state.index.ntotal,
        "index_file": index_name,
        "index_sha256": sha256,
        "docs": [
            {"id": internal_id, "chunk": _chunk_to_json(chunk)}
            for internal_id, chunk in state.docs.items()
        ],
    }
    _replace_from_temp(
        directory, _DOCSTORE_FILENAME, json.dumps(docstore).encode("utf-8")
    )
    state.index_file = index_name
    _after_commit(directory, keep=index_name)


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
        self._states: dict[str | None, _IndexState] = {}
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

    def _state(self, key: str | None) -> _IndexState | None:
        state = self._states.get(key)
        if state is None and key is not None:
            state = _read_state(Path(key))
            if state is not None:
                self._states[key] = state
        return state

    def _persist(self, key: str | None, state: _IndexState) -> None:
        if key is not None:
            _write_state(Path(key), state)

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
                    _chunk_to_json(doc)
            try:
                state = self._state(key)
                if state is None:
                    dimension = len(vectors[0])
                    state = _IndexState(
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
