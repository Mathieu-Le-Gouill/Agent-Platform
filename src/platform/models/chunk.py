
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from models.enums.language import Language
from models.enums.fileformat import FileFormat


@dataclass(slots=True, frozen=True)
class ChunkMetadata:
    source: str | None = None
    format: FileFormat | None = None
    language: Language | None = None
 
    extra: dict[str, Any] = field(default_factory=dict)
 

@dataclass(slots=True, frozen=True)
class Chunk:
    document_id: UUID
    text: str
    index: int
    
    id: UUID = field(default_factory=uuid4)
    start_char: int | None = None
    end_char: int | None = None

    # Retrieval metadata
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)
