from __future__ import annotations

from typing import Any
from uuid import UUID

from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.enums import Language

__all__ = ["chunk_to_payload", "point_to_chunk"]


def chunk_to_payload(chunk: TextChunk) -> dict[str, Any]:
    metadata = chunk.metadata or {}
    return {
        "text": chunk.text,
        "document_id": str(chunk.document_id) if chunk.document_id else None,
        "index": chunk.index,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        "format": chunk.format.value if chunk.format else None,
        "source": metadata.get("source"),
        "language": metadata.get("language"),
        "extra": metadata.get("extra"),
    }


def point_to_chunk(point: Any) -> TextChunk:
    payload = point.payload or {}
    return TextChunk(
        id=UUID(str(point.id)),
        document_id=UUID(payload["document_id"])
        if payload.get("document_id")
        else None,
        text=payload.get("text") or "",
        index=payload.get("index") or 0,
        start_char=payload.get("start_char"),
        end_char=payload.get("end_char"),
        metadata={
            "source": payload.get("source"),
            "language": Language(payload["language"])
            if payload.get("language")
            else None,
            "extra": payload.get("extra") or {},
        },
    )
