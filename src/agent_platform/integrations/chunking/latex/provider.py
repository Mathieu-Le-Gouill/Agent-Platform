from collections.abc import Sequence

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_text_splitters.base import TextSplitter

from agent_platform.core.errors import ValidationError
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.document import TextDocument
from agent_platform.core.schemas.enums import DocumentFormat
from agent_platform.integrations.chunking.langchain_base import LangChainChunker
from agent_platform.integrations.chunking.latex.config import LatexChunkerConfig

_ACCEPTED_FORMATS = {DocumentFormat.LATEX, DocumentFormat.UNKNOWN}


class LatexChunkerProvider(LangChainChunker[LatexChunkerConfig]):
    def _default_config(self) -> LatexChunkerConfig:
        return LatexChunkerConfig()

    def chunk(
        self,
        documents: Sequence[TextDocument],
        config: LatexChunkerConfig | None = None,
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
        # Note: is_separator_regex is intentionally not forwarded here —
        # from_language() always calls cls(..., is_separator_regex=True, **kwargs),
        # so passing it again would raise a duplicate-kwarg TypeError. The field
        # is still exposed on the config for documentation/consistency with
        # `recursive`, but LaTeX separators are always treated as regex.
        return RecursiveCharacterTextSplitter.from_language(
            Language.LATEX,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            add_start_index=config.add_start_index,
            keep_separator=config.keep_separator,
            strip_whitespace=config.strip_whitespace,
        )
