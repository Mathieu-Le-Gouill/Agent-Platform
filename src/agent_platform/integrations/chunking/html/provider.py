from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

from langchain_text_splitters import (
    HTMLHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from agent_platform.core.errors import ProviderError, ValidationError
from agent_platform.core.interfaces.chunking.base import BaseChunker
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.document import TextDocument
from agent_platform.core.schemas.enums import DocumentFormat
from agent_platform.integrations.chunking.html.config import HTMLChunkerConfig

_ACCEPTED_FORMATS = {DocumentFormat.HTML, DocumentFormat.UNKNOWN}


class HTMLStructureChunkerProvider(
    BaseChunker[TextDocument, TextChunk, HTMLChunkerConfig]
):
    def _default_config(self) -> HTMLChunkerConfig:
        return HTMLChunkerConfig()

    def chunk(
        self,
        documents: Sequence[TextDocument],
        config: HTMLChunkerConfig | None = None,
    ) -> list[TextChunk]:
        config = config or self._default_config()

        header_splitter = HTMLHeaderTextSplitter(
            headers_to_split_on=config.headers_to_split_on,
            return_each_element=config.return_each_element,
        )
        # NOTE: unlike `recursive`/`latex`, this provider splits at the string
        # level (split_text) rather than going through LangChainChunker's
        # split_documents path, so start_char/end_char are never populated on
        # the resulting TextChunks. Left as-is per architecture review — see
        # `markdown.py` for the same tradeoff.
        size_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

        chunks: list[TextChunk] = []
        for doc in documents:
            if doc.format not in _ACCEPTED_FORMATS:
                raise ValidationError(
                    f"HTMLStructureChunkerProvider expects DocumentFormat.HTML "
                    f"(or UNKNOWN), got {doc.format}"
                )

            try:
                sections = header_splitter.split_text(doc.text)
            except ImportError as exc:
                raise ProviderError(str(exc)) from exc

            index = 0
            for section in sections:
                for piece in size_splitter.split_text(section.page_content):
                    if not piece.strip():
                        continue
                    chunks.append(
                        TextChunk(
                            id=uuid4(),
                            document_id=doc.id,
                            text=piece,
                            index=index,
                            format=DocumentFormat.HTML,
                            metadata={
                                "source": doc.source,
                                "headers": dict(section.metadata),
                            },
                        )
                    )
                    index += 1
        return chunks
