from agent_platform.core.entities.text.chunk import Chunk
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any
from uuid import UUID, uuid4

   

@dataclass(slots=True)
class Document:
    id: UUID = field(default_factory=uuid4)

    # Source information
    source: str = ""
    title: Optional[str] = None
    document_type: Optional[str] = None
    url: Optional[str] = None

    # Global information
    language: Optional[str] = None
    summary: Optional[str] = None

    # Classification
    tags: list[str] = field(default_factory=list)

    # Lifecycle
    created_at: datetime = field(default_factory=datetime.now)

    # Arbitrary source metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Chunks
    chunks: list["Chunk"] = field(default_factory=list)


    # --- Constructors ---

    @classmethod
    def from_url(
        cls,
        url: str,
    ) -> "Document":
        ...

    @classmethod
    def from_files(
        cls,
        file_path: str,
        file_type: str
    ) -> "Document":
        ...


    # --- Tools ---