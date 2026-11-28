from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import faiss
import numpy as np
from pydantic_core import PydanticSerializationError

from agent_platform.core.errors import ConfigError, ValidationError
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.integrations.vector_store.faiss.config import DistanceMetric

_INDEX_FILE_PATTERN = re.compile(r"index-[0-9a-f]{64}\.faiss")
_TEMP_FILE_PATTERN = re.compile(
    r"\.(docstore\.json|index-[0-9a-f]{64}\.faiss)\.[0-9a-f]{32}\.tmp"
)
# Named after the provider module this code moved out of, so existing log
# filters on that name keep receiving the cleanup warning.
logger = logging.getLogger("agent_platform.integrations.vector_store.faiss.provider")
_DOCSTORE_FILENAME = "docstore.json"
_LEGACY_DOCSTORE_FILENAME = "docstore.pkl"
_DOCSTORE_VERSION = 1
_REINGEST = "delete the directory and re-ingest its documents"

# Written as-is: the serialized index is a uint8 array, never copied to bytes.
Buffer = bytes | np.ndarray


@dataclass
class IndexState:
    index: faiss.IndexIDMap2
    dimension: int
    distance: DistanceMetric
    docs: dict[int, TextChunk] = field(default_factory=dict)
    uuid_to_id: dict[UUID, int] = field(default_factory=dict)
    next_id: int = 0
    # name of the index file the committed docstore.json references, if any
    index_file: str | None = None


def chunk_to_json(chunk: TextChunk) -> dict[str, Any]:
    try:
        return chunk.model_dump(mode="json")
    except PydanticSerializationError as exc:
        raise ValidationError(
            f"FAISS stores chunk metadata as JSON; chunk {chunk.id} has a value "
            f"JSON cannot hold: {exc}"
        ) from exc


def _index_filename(sha256: str) -> str:
    return f"index-{sha256}.faiss"


def read_state(directory: Path) -> IndexState | None:
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

    return IndexState(
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


def write_state(directory: Path, state: IndexState) -> None:
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
            {"id": internal_id, "chunk": chunk_to_json(chunk)}
            for internal_id, chunk in state.docs.items()
        ],
    }
    _replace_from_temp(
        directory, _DOCSTORE_FILENAME, json.dumps(docstore).encode("utf-8")
    )
    state.index_file = index_name
    _after_commit(directory, keep=index_name)
