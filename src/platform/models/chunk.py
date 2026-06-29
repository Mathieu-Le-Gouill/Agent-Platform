
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from models.score import Score, ScoreKind
from models.language import Language
from models.document import DocumentType

@dataclass(slots=True, frozen=True)
class CoordinatesMetadata:
    """Bounding-box / polygon coordinates as returned by Unstructured."""
    points: list[tuple[float, float]] | None = None
    system: str | None = None  # e.g. "PixelSpace", "RelativeCoordinateSystem"


@dataclass(slots=True, frozen=True)
class ChunkMetadata:
    # --- Document-level context ---
    source: str | None = None
    title: str | None = None
    document_type: DocumentType | None = None
    language: Language | None = None
 
    # --- Element / chunk position ---
    page_number: int | None = None
    page_name: str | None = None      # slide name (PPTX), sheet name (XLSX), etc.
    section: str | None = None        # nearest ancestor heading / parent element text
    category_depth: int | None = None # heading depth (0 = top-level Title)

    # --- Element identity ---
    element_id: str | None = None
    parent_id: str | None = None      # id of the parent element (e.g. table → row)
    category: str | None = None       # Unstructured category: Title, NarrativeText, Table…
    is_continuation: bool | None = None

    # --- Rich inline content ---
    text_as_html: str | None = None                     # HTML rendering of tables
    links: list[dict[str, Any]] | None = None           # [{text, url, start_index}]
    link_texts: list[str] | None = None
    link_urls: list[str] | None = None
    emphasized_text_contents: list[str] | None = None
    emphasized_text_tags: list[str] | None = None       # ["b", "i", …]

    # --- Image elements ---
    image_base64: str | None = None
    image_mime_type: str | None = None
    image_path: str | None = None
    image_url: str | None = None
 
    # --- Audio / video segments ---
    segment_start_seconds: float | None = None
    segment_end_seconds: float | None = None
 
    # --- Email-specific fields ---
    email_message_id: str | None = None
    sent_from: list[str] | None = None
    sent_to: list[str] | None = None
    cc_recipient: list[str] | None = None
    bcc_recipient: list[str] | None = None
    subject: str | None = None        # email subject (also used for doc subject)
    signature: str | None = None

    # --- Table metadata ---
    table_id: str | None = None
    table_as_cells: dict[str, Any] | None = None
    table_extraction_method: str | None = None

    # --- Spatial layout ---
    coordinates: CoordinatesMetadata | None = None
 
    # --- Arbitrary overflow ---
    extra: dict[str, Any] = field(default_factory=dict)
 


@dataclass(slots=True)
class Chunk:
    document_id: UUID
    
    id: UUID = field(default_factory=uuid4)

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
