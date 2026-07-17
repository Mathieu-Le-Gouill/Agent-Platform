from typing import Sequence

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_text_splitters.base import TextSplitter

from agent_platform.integrations.chunking.langchain_base import LangChainChunker
from agent_platform.integrations.chunking.latex.config import LatexChunkerConfig
from agent_platform.core.errors import ValidationError
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.document import TextDocument
from agent_platform.core.schemas.enums import DocumentFormat

_ACCEPTED_FORMATS = {DocumentFormat.LATEX, DocumentFormat.UNKNOWN}


class LatexChunkerProvider(LangChainChunker[LatexChunkerConfig]):
    def _default_config(self) -> LatexChunkerConfig:
        return LatexChunkerConfig()

    def chunk(
        self,
        documents: Sequence[TextDocument],
        config: LatexChunkerConfig | None,
    ) -> list[TextChunk]:
        for doc in documents:
            if doc.format not in _ACCEPTED_FORMATS:
                raise ValidationError(
                    f"LatexChunkerProvider expects DocumentFormat.LATEX "
                    f"(or UNKNOWN), got {doc.format}"
                )
        return super().chunk(documents, config)

    def _splitter(
        self,
        config: LatexChunkerConfig,
    ) -> TextSplitter:
        return RecursiveCharacterTextSplitter.from_language(
            Language.LATEX,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            add_start_index=config.add_start_index,
        )
