from typing import Any

from langchain_core.documents import BaseDocumentCompressor
from langchain_voyageai import VoyageAIRerank

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
        kwargs: dict[str, Any] = {}
        if config.truncation is not None:
            kwargs["truncation"] = config.truncation
        if self._credentials.api_key is not None:
            kwargs["api_key"] = self._credentials.api_key

        return VoyageAIRerank(
            model=config.model,
            top_k=config.top_k,
            **kwargs,
        )
