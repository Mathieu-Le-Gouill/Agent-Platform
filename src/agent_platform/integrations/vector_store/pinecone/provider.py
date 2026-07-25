from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore

from agent_platform.core.credentials import resolve_credentials
from agent_platform.core.errors import require_secret
from agent_platform.integrations.credentials import PineconeCredentials
from agent_platform.integrations.vector_store.langchain_base import LangChainVectorStore
from agent_platform.integrations.vector_store.pinecone.config import PineconeConfig


class PineconeStore(LangChainVectorStore[PineconeConfig]):
    def __init__(
        self,
        credentials: PineconeCredentials | None = None,
        embeddings: Embeddings | None = None,
    ) -> None:
        super().__init__(embeddings)
        self._credentials = resolve_credentials(credentials, PineconeCredentials)

    def _default_config(self) -> PineconeConfig:
        return PineconeConfig()

    def _build_client(self, config: PineconeConfig) -> PineconeVectorStore:
        api_key = require_secret(
            self._credentials.api_key, "Pinecone API key is required"
        )
        kwargs: dict = {
            "index_name": config.collection_name,
            "embedding": self._embeddings,
            "pinecone_api_key": api_key.get_secret_value(),
            "namespace": config.namespace,
        }
        if config.host is not None:
            kwargs["host"] = config.host
        return PineconeVectorStore(**kwargs)
