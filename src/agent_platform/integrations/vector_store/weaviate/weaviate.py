from __future__ import annotations

import weaviate
import weaviate.auth
from langchain_weaviate import WeaviateVectorStore
from langchain_core.embeddings import Embeddings

from agent_platform.integrations.credentials.weaviate import WeaviateCredentials
from agent_platform.integrations.vector_store.langchain_base import LangChainVectorStore
from agent_platform.integrations.vector_store.weaviate.config import WeaviateConfig


class WeaviateStore(LangChainVectorStore[WeaviateCredentials, WeaviateConfig]):
    def __init__(
        self,
        credentials: WeaviateCredentials | None = None,
        embeddings: Embeddings | None = None,
    ) -> None:
        super().__init__(
            credentials if credentials is not None else WeaviateCredentials(),
            embeddings,
        )

    def _default_config(self) -> WeaviateConfig:
        return WeaviateConfig()

    def _build_client(self, config: WeaviateConfig) -> WeaviateVectorStore:
        if self._credentials.api_key:
            client = weaviate.connect_to_custom(
                http_host=self._credentials.url.split("://")[-1].split(":")[0],
                http_port=int(self._credentials.url.split(":")[-1])
                if ":" in self._credentials.url.split("://")[-1]
                else 443,
                http_secure=self._credentials.url.startswith("https"),
                grpc_host=self._credentials.url.split("://")[-1].split(":")[0],
                grpc_port=50051,
                grpc_secure=self._credentials.url.startswith("https"),
                auth_credentials=weaviate.auth.AuthApiKey(
                    self._credentials.api_key.get_secret_value()
                ),
            )
        else:
            client = weaviate.connect_to_local(url=self._credentials.url)

        return WeaviateVectorStore(
            client=client,
            index_name=config.collection_name,
            text_key=config.text_key,
            embedding=self._embeddings,
        )
