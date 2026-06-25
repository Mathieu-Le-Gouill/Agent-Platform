from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any
from uuid import UUID, uuid4
from models.chunk import Chunk
from models.language import Language


@dataclass(slots=True, frozen=True)
class DocumentMetadata:
    author:      Optional[str] = None
    description: Optional[str] = None
    created_at:  Optional[datetime] = None   # source system's date, distinct from your lifecycle created_at
    modified_at: Optional[datetime] = None
    extra:       dict[str, Any] = field(default_factory=dict)
   

@dataclass(slots=True)
class Document:
    id: UUID = field(default_factory=uuid4)

    # Source
    source: str = ""
    title: Optional[str] = None
    document_type: Optional[str] = None
    url: Optional[str] = None

    # Content
    text: Optional[str] = None

    # Global information
    language: Optional[Language] = None
    tags: list[str] = field(default_factory=list)

    # Lifecycle
    created_at: datetime = field(default_factory=datetime.now)

    # Arbitrary source metadata
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)

    # Chunks
    chunks: list[Chunk] = field(default_factory=list)