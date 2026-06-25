
from dataclasses import dataclass, field
from typing import Optional, Any
from uuid import UUID, uuid4
from models.score import Score, ScoreKind
from models.language import Language


@dataclass(slots=True, frozen=True)
class ChunkMetadata:
    # Document parent data
    source: Optional[str] = None
    title: Optional[str] = None
    document_type: Optional[str] = None
    language:Optional[Language] = None

    # Chunck specific
    page_number: Optional[int] = None
    section: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    id: UUID = field(default_factory=uuid4)

    document_id: Optional[UUID] = None
    text: str = ""
    index: int = 0
    start_char: Optional[int] = None
    end_char: Optional[int] = None

    # Retrieval metadata
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)

    # Embedding (optional cache)
    embedding: Optional[list[float]] = None
    embedding_model: Optional[str] = None

    # RAG retrieval and classification tasks scores
    scores: dict[ScoreKind, Score] = field(default_factory=dict)

    summaries: Optional[list[str]] = None

    translations: Optional[dict[Language, str]] = None
