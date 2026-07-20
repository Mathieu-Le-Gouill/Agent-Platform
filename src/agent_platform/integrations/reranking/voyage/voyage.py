from langchain_voyageai import VoyageAIRerank
from langchain_core.documents import BaseDocumentCompressor

from agent_platform.integrations.credentials import VoyageCredentials
from agent_platform.integrations.reranking.langchain_base import LangChainReranker
from agent_platform.integrations.reranking.voyage.config import VoyageRerankerConfig


class VoyageRerankerProvider(LangChainReranker[VoyageRerankerConfig]):
    def __init__(self, credentials: VoyageCredentials | None = None) -> None:
        self._credentials = (
            credentials if credentials is not None else VoyageCredentials()
        )

    def _default_config(self) -> VoyageRerankerConfig:
        return VoyageRerankerConfig()

    def _client(self, config: VoyageRerankerConfig) -> BaseDocumentCompressor:
        api_key = (
            self._credentials.api_key.get_secret_value()
            if self._credentials.api_key
            else None
        )
        kwargs = {}
        if config.truncation is not None:
            kwargs["truncation"] = config.truncation

        return VoyageAIRerank(
            model=config.model,
            voyage_api_key=api_key,
            top_k=config.top_k,
            **kwargs,
        )
