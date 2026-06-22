from agent_platform.core.entities.text.chunk import Chunk
from agent_platform.core.entities.content import Content
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class DocumentMetadata:
    source: str | None = None
    source_type: str | None = None
    author: str | None = None
    language: str | None = None
    created_at: datetime | None = None
    tags: list[str] = field(default_factory=list)
    extra: dict[str, str | int | float | bool] = field(default_factory=dict)


@dataclass(slots=True)
class Document(Content):
    id: str
    content: str
    metadata: DocumentMetadata | None

    # --- Constructors ---

    def __init__(
        self,
        id: str,
        content: str,
        metadata: DocumentMetadata | None = None,
    ) -> None:
        
        self.id = id
        self.content = content
        self.metadata = metadata


    @classmethod
    def from_url(
        cls,
        url: str,
    ) -> "Document":
        ...


    def split(
        self,
        chunck_size: int
    ) -> list[Chunk]:
        ...