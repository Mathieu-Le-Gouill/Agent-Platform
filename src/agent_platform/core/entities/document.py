from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any
from uuid import UUID, uuid4
from agent_platform.core.entities.chunk import Chunk
from core.value_objects.language import Language
   

@dataclass(slots=True)
class Document:
    id: UUID = field(default_factory=uuid4)

    # Source
    source: str = ""
    title: Optional[str] = None
    document_type: Optional[str] = None
    url: Optional[str] = None

    # Content
    raw_text: Optional[str] = None

    # Global information
    language: Optional[Language] = None
    tags: list[str] = field(default_factory=list)

    # Lifecycle
    created_at: datetime = field(default_factory=datetime.now)

    # Arbitrary source metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Chunks
    chunks: list[Chunk] = field(default_factory=list)