
from dataclasses import dataclass, field
from typing import Optional, Any
from uuid import UUID, uuid4

@dataclass(slots=True)
class Chunk:
    id: UUID = field(default_factory=uuid4)

    document_id: Optional[UUID] = None
    content: str = ""

    # Position within the document
    index: int = 0
    page_number: Optional[int] = None
    section: Optional[str] = None

    # Retrieval metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Embedding (optional cache)
    embedding: Optional[list[float]] = None
    embedding_model: Optional[str] = None
