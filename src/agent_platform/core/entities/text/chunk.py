
from dataclasses import dataclass, field
from agent_platform.core.entities.content import Content
from typing import Optional, Any
from uuid import UUID, uuid4

@dataclass(slots=True)
class Chunk(Content):
    id: UUID = field(default_factory=uuid4)

    document_id: Optional[UUID] = None
    content: str = ""

    # Position
    index: int = 0
    page_number: Optional[int] = None
    section: Optional[str] = None

    # Optional chunk-level enrichment
    summary: Optional[str] = None

    # Retrieval metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Embedding (optional cache)
    embedding: Optional[list[float]] = None
    embedding_model: Optional[str] = None
