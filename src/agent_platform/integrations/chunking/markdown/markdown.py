from __future__ import annotations

from typing import Sequence
from uuid import uuid4

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from agent_platform.core.interfaces.chunking.base import BaseChunker
from agent_platform.core.errors import ValidationError
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.document import TextDocument
from agent_platform.core.schemas.enums import DocumentFormat
from agent_platform.integrations.chunking.markdown.config import MarkdownChunkerConfig

_ACCEPTED_FORMATS = {DocumentFormat.MARKDOWN, DocumentFormat.UNKNOWN}


class MarkdownStructureChunkerProvider(
    BaseChunker[TextDocument, TextChunk, MarkdownChunkerConfig]
):
    def _default_config(self) -> MarkdownChunkerConfig:
        return MarkdownChunkerConfig()

    def chunk(
        self,
        documents: Sequence[TextDocument],
        config: MarkdownChunkerConfig | None,
    ) -> list[TextChunk]:
        config = config or self._default_config()

        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=config.headers_to_split_on,
            strip_headers=config.strip_headers,
        )
        size_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

        chunks: list[TextChunk] = []
        for doc in documents:
            if doc.format not in _ACCEPTED_FORMATS:
                raise ValidationError(
                    f"MarkdownStructureChunkerProvider expects "
                    f"DocumentFormat.MARKDOWN (or UNKNOWN), got {doc.format}"
                )

            index = 0
            for section in header_splitter.split_text(doc.text):
                for piece in size_splitter.split_text(section.page_content):
                    if not piece.strip():
                        continue
                    chunks.append(
                        TextChunk(
                            id=uuid4(),
                            document_id=doc.id,
                            text=piece,
                            index=index,
                            format=DocumentFormat.MARKDOWN,
                            metadata={
                                "source": doc.source,
                                "headers": dict(section.metadata),
                            },
                        )
                    )
                    index += 1
        return chunks
