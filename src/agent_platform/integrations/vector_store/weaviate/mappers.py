from __future__ import annotations

from typing import Any
from uuid import UUID

from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.enums import Language

__all__ = ["chunk_to_properties", "object_to_chunk"]


def chunk_to_properties(chunk: TextChunk, text_key: str) -> dict[str, Any]:
    metadata = chunk.metadata or {}
    return {
        text_key: chunk.text,
        "document_id": str(chunk.document_id) if chunk.document_id else None,
        "index": chunk.index,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        "format": chunk.format.value if chunk.format else None,
        "source": metadata.get("source"),
        "language": metadata.get("language"),
        "extra": metadata.get("extra"),
    }


def object_to_chunk(obj: Any, text_key: str) -> TextChunk:
    props = obj.properties
    return TextChunk(
        id=obj.uuid,
        document_id=UUID(props["document_id"]) if props.get("document_id") else None,
        text=props.get(text_key) or "",
        index=props.get("index") or 0,
        start_char=props.get("start_char"),
        end_char=props.get("end_char"),
        metadata={
            "source": props.get("source"),
            "language": Language(props["language"]) if props.get("language") else None,
            "extra": props.get("extra") or {},
        },
    )
