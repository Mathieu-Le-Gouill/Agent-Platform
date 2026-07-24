from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.base import TextSplitter

from agent_platform.integrations.chunking.langchain_base import LangChainChunker
from agent_platform.integrations.chunking.recursive.config import RecursiveChunkerConfig


class RecursiveChunkerProvider(LangChainChunker[RecursiveChunkerConfig]):
    def _default_config(self) -> RecursiveChunkerConfig:
        return RecursiveChunkerConfig()

    def _splitter(
        self,
        config: RecursiveChunkerConfig,
    ) -> TextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            add_start_index=config.add_start_index,
            separators=config.separators,
            keep_separator=config.keep_separator,
            is_separator_regex=config.is_separator_regex,
            strip_whitespace=config.strip_whitespace,
        )
