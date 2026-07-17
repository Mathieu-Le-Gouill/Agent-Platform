from langchain_cohere import CohereRerank

from agent_platform.integrations.credentials import CohereCredentials
from agent_platform.integrations.reranking.langchain_base import LangChainReranker
from agent_platform.integrations.reranking.cohere.config import CohereRerankerConfig


class CohereRerankerProvider(LangChainReranker[CohereRerankerConfig]):
    def __init__(self, credentials: CohereCredentials | None = None) -> None:
        self._credentials = (
            credentials if credentials is not None else CohereCredentials()
        )

    def _default_config(self) -> CohereRerankerConfig:
        return CohereRerankerConfig()

    def _client(self, config: CohereRerankerConfig) -> CohereRerank:
        return CohereRerank(
            model=config.model,
            cohere_api_key=self._credentials.api_key.get_secret_value()
            if self._credentials.api_key
            else None,
        )
