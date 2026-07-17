from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from agent_platform.integrations.vector_store.langchain_base import LangChainVectorStore
from agent_platform.integrations.vector_store.chroma.config import ChromaConfig


class ChromaStore(LangChainVectorStore[ChromaConfig]):
    def _default_config(self) -> ChromaConfig:
        return ChromaConfig()

    def _build_client(self, config: ChromaConfig) -> Chroma:
        return Chroma(
            collection_name=config.collection_name,
            embedding_function=self._embeddings,
            host=config.host,
            port=config.port,
        )
