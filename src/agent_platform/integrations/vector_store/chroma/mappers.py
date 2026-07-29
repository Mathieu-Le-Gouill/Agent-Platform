from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.enums import Language

__all__ = ["chunk_to_metadata", "row_to_chunk"]


def chunk_to_metadata(chunk: TextChunk) -> dict[str, Any]:
    metadata = chunk.metadata or {}
    result: dict[str, Any] = {"extra": json.dumps(metadata.get("extra") or {})}
    if chunk.document_id:
        result["document_id"] = str(chunk.document_id)
    if chunk.index is not None:
        result["index"] = chunk.index
    if chunk.start_char is not None:
        result["start_char"] = chunk.start_char
    if chunk.end_char is not None:
        result["end_char"] = chunk.end_char
    if chunk.format:
        result["format"] = chunk.format.value
    if metadata.get("source") is not None:
        result["source"] = metadata["source"]
    if metadata.get("language") is not None:
        result["language"] = metadata["language"]
    return result


def row_to_chunk(
    id_: str, text: str | None, metadata: dict[str, Any] | None
) -> TextChunk:
    metadata = metadata or {}
    extra: dict[str, Any] = {}
    if metadata.get("extra"):
        try:
            extra = json.loads(metadata["extra"])
        except (TypeError, ValueError):
            extra = {}
    return TextChunk(
        id=UUID(id_),
        document_id=UUID(metadata["document_id"])
        if metadata.get("document_id")
        else None,
        text=text or "",
        index=metadata.get("index") or 0,
        start_char=metadata.get("start_char"),
        end_char=metadata.get("end_char"),
        metadata={
            "source": metadata.get("source"),
            "language": Language(metadata["language"])
            if metadata.get("language")
            else None,
            "extra": extra,
        },
    )
