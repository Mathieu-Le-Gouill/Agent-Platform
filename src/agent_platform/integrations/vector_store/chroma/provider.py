from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from agent_platform.core.credentials import resolve_credentials
from agent_platform.integrations.credentials import ChromaCredentials
from agent_platform.integrations.vector_store.chroma.config import ChromaConfig
from agent_platform.integrations.vector_store.langchain_base import LangChainVectorStore


class ChromaStore(LangChainVectorStore[ChromaConfig]):
    def __init__(
        self,
        credentials: ChromaCredentials | None = None,
        embeddings: Embeddings | None = None,
    ) -> None:
        super().__init__(embeddings)
        self._credentials = resolve_credentials(credentials, ChromaCredentials)

    def _default_config(self) -> ChromaConfig:
        return ChromaConfig()

    def _build_client(self, config: ChromaConfig) -> Chroma:
        kwargs: dict = {
            "collection_name": config.collection_name,
            "embedding_function": self._embeddings,
            "tenant": config.tenant,
            "database": config.database,
        }
        if self._credentials.api_key:
            kwargs["chroma_cloud_api_key"] = (
                self._credentials.api_key.get_secret_value()
            )

        if config.persist_directory is not None:
            kwargs["persist_directory"] = config.persist_directory
        else:
            kwargs["host"] = config.host
            kwargs["port"] = config.port
            kwargs["ssl"] = config.ssl

        return Chroma(**kwargs)
