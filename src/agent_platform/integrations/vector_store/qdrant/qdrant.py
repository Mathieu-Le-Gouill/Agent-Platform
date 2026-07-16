from langchain_qdrant import QdrantVectorStore
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient

from agent_platform.integrations.credentials.qdrant import QdrantCredentials
from agent_platform.integrations.vector_store.langchain_base import LangChainVectorStore
from agent_platform.integrations.vector_store.qdrant.config import QdrantConfig


class QdrantVectorStoreProvider(LangChainVectorStore[QdrantCredentials, QdrantConfig]):
    def __init__(
        self,
        credentials: QdrantCredentials | None = None,
        embeddings: Embeddings | None = None,
    ) -> None:
        super().__init__(
            credentials if credentials is not None else QdrantCredentials(), embeddings
        )

    def _default_config(self) -> QdrantConfig:
        return QdrantConfig()

    def _build_client(self, config: QdrantConfig) -> QdrantVectorStore:
        client = QdrantClient(
            url=config.url,
            api_key=self._credentials.api_key.get_secret_value()
            if self._credentials.api_key
            else None,
        )
        return QdrantVectorStore(
            client=client,
            collection_name=config.collection_name,
            embedding=self._embeddings,
        )
