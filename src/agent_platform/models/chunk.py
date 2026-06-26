
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from models.score import Score, ScoreKind
from models.language import Language


@dataclass(slots=True, frozen=True)
class ChunkMetadata:
    # Document parent data
    source: str | None = None
    title: str | None = None
    document_type: str | None = None
    language: Language | None = None

    # Chunck specific
    page_number: int | None = None
    section: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    id: UUID = field(default_factory=uuid4)

    document_id: UUID | None = None
    text: str = ""
    index: int = 0
    start_char: int | None = None
    end_char: int | None = None

    # Retrieval metadata
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)

    # Embedding (optional cache)
    embedding: list[float] | None = None
    embedding_model: str | None = None

    # RAG retrieval and classification tasks scores
    scores: dict[ScoreKind, Score] = field(default_factory=dict)

    summaries: list[str] | None = None

    translations: dict[Language, str] | None = None
