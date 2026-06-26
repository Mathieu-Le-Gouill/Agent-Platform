from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from models.chunk import Chunk
from models.language import Language


@dataclass(slots=True, frozen=True)
class DocumentMetadata:
    author:      str | None = None
    description: str | None = None
    created_at:  datetime | None = None   # source system's date, distinct from your lifecycle created_at
    modified_at: datetime | None = None
    extra:       dict[str, Any] = field(default_factory=dict)
   

@dataclass(slots=True)
class Document:
    id: UUID = field(default_factory=uuid4)

    # Source
    source: str = ""
    title: str | None = None
    document_type: str | None = None
    url: str | None = None

    # Content
    text: str | None = None

    # Global information
    language: Language | None = None
    tags: list[str] = field(default_factory=list)

    # Lifecycle
    created_at: datetime = field(default_factory=datetime.now)

    # Arbitrary source metadata
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)

    # Chunks
    chunks: list[Chunk] = field(default_factory=list)