
from dataclasses import dataclass, field
from typing import Optional, Any
from uuid import UUID, uuid4


@dataclass(slots=True, frozen=True)
class ChunkMetadata:
    source:        Optional[str] = None
    title:         Optional[str] = None
    document_type: Optional[str] = None
    language:      Optional[str] = None
    page_number:   Optional[int] = None
    section:       Optional[str] = None
    extra:         dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    id: UUID = field(default_factory=uuid4)

    document_id: Optional[UUID] = None
    content: str = ""
    index: int = 0

    # Retrieval metadata
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)

    # Embedding (optional cache)
    embedding: Optional[list[float]] = None
    embedding_model: Optional[str] = None
