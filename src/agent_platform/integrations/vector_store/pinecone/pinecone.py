from langchain_pinecone import PineconeVectorStore
from langchain_core.embeddings import Embeddings

from agent_platform.integrations.credentials.pinecone import PineconeCredentials
from agent_platform.integrations.vector_store.langchain_base import LangChainVectorStore
from agent_platform.integrations.vector_store.pinecone.config import PineconeConfig
from agent_platform.core.errors import MissingCredentialError


class PineconeStore(LangChainVectorStore[PineconeCredentials, PineconeConfig]):
    def __init__(
        self,
        credentials: PineconeCredentials | None = None,
        embeddings: Embeddings | None = None,
    ) -> None:
        super().__init__(
            credentials if credentials is not None else PineconeCredentials(),
            embeddings,
        )

    def _default_config(self) -> PineconeConfig:
        return PineconeConfig()

    def _build_client(self, config: PineconeConfig) -> PineconeVectorStore:
        if not self._credentials.api_key:
            raise MissingCredentialError("Pinecone API key is required")
        return PineconeVectorStore(
            index_name=config.collection_name,
            embedding=self._embeddings,
            pinecone_api_key=self._credentials.api_key.get_secret_value(),
            namespace=config.namespace,
        )
