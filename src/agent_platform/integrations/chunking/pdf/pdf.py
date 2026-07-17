from __future__ import annotations

from typing import Sequence
from uuid import uuid4

from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf

from agent_platform.core.interfaces.chunking.base import BaseChunker
from agent_platform.core.errors import ProviderError, ValidationError
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.document import TextDocument
from agent_platform.core.schemas.enums import DocumentFormat
from agent_platform.integrations.chunking.pdf.config import PDFChunkerConfig

_ACCEPTED_FORMATS = {DocumentFormat.PDF, DocumentFormat.UNKNOWN}


class PDFStructureChunkerProvider(
    BaseChunker[TextDocument, TextChunk, PDFChunkerConfig]
):
    def _default_config(self) -> PDFChunkerConfig:
        return PDFChunkerConfig()

    def chunk(
        self,
        documents: Sequence[TextDocument],
        config: PDFChunkerConfig | None,
    ) -> list[TextChunk]:
        config = config or self._default_config()

        chunks: list[TextChunk] = []
        for doc in documents:
            if doc.format not in _ACCEPTED_FORMATS:
                raise ValidationError(
                    f"PDFStructureChunkerProvider expects DocumentFormat.PDF "
                    f"(or UNKNOWN), got {doc.format}"
                )

            if not doc.source:
                raise ValidationError(
                    "PDFStructureChunkerProvider requires TextDocument.source "
                    "to be a path to the source PDF file"
                )

            try:
                elements = partition_pdf(filename=doc.source)
                sections = chunk_by_title(
                    elements,
                    max_characters=config.chunk_size,
                    overlap=config.chunk_overlap,
                    combine_text_under_n_chars=config.combine_text_under_n_chars,
                    new_after_n_chars=config.new_after_n_chars,
                    multipage_sections=config.multipage_sections,
                )
            except ValidationError:
                raise
            except Exception as exc:
                raise ProviderError(f"Failed to chunk PDF {doc.source}: {exc}") from exc

            for index, section in enumerate(sections):
                metadata = (
                    section.metadata.to_dict()
                    if hasattr(section.metadata, "to_dict")
                    else {}
                )
                chunks.append(
                    TextChunk(
                        id=uuid4(),
                        document_id=doc.id,
                        text=str(section),
                        index=index,
                        format=DocumentFormat.PDF,
                        metadata={
                            "source": doc.source,
                            "page_number": metadata.get("page_number"),
                            "element_category": type(section).__name__,
                        },
                    )
                )
        return chunks
