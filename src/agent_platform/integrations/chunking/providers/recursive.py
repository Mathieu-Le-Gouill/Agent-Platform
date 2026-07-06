from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent_platform.integrations.chunking.langchain_base import LangChainChunker
from agent_platform.integrations.chunking.config import ChunkerConfig


class RecursiveChunkerProvider(LangChainChunker):
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        add_start_index: bool = False,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.add_start_index = add_start_index

    def _splitter(
        self,
        config: ChunkerConfig | None,
    ):

        return RecursiveCharacterTextSplitter(
            chunk_size=(config.chunk_size if config else self.chunk_size),
            chunk_overlap=(config.chunk_overlap if config else self.chunk_overlap),
            add_start_index=True,
        )
