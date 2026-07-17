from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent_platform.integrations.chunking.langchain_base import LangChainChunker
from agent_platform.integrations.chunking.recursive.config import RecursiveChunkerConfig
from langchain_text_splitters.base import TextSplitter


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
        )
