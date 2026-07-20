from typing import Any

from langchain_qdrant import QdrantVectorStore
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient, models

from agent_platform.integrations.credentials import QdrantCredentials
from agent_platform.integrations.vector_store.langchain_base import LangChainVectorStore
from agent_platform.integrations.vector_store.qdrant.config import QdrantConfig


class QdrantVectorStoreProvider(LangChainVectorStore[QdrantConfig]):
    def __init__(
        self,
        credentials: QdrantCredentials | None = None,
        embeddings: Embeddings | None = None,
    ) -> None:
        super().__init__(embeddings)
        self._credentials = (
            credentials if credentials is not None else QdrantCredentials()
        )

    def _default_config(self) -> QdrantConfig:
        return QdrantConfig()

    def _build_client(self, config: QdrantConfig) -> QdrantVectorStore:
        client = QdrantClient(
            url=config.url,
            api_key=self._credentials.api_key.get_secret_value()
            if self._credentials.api_key
            else None,
            prefer_grpc=config.prefer_grpc,
        )
        return QdrantVectorStore(
            client=client,
            collection_name=config.collection_name,
            embedding=self._embeddings,
        )

    def _search_kwargs(
        self, config: QdrantConfig, filter: dict[str, Any] | None
    ) -> dict[str, Any]:
        if not filter:
            return {"filter": None}
        return {
            "filter": models.Filter(
                must=[
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                    for key, value in filter.items()
                ]
            )
        }
