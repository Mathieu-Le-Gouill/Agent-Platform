from langchain_cohere import CohereRerank

from agent_platform.integrations.credentials import CohereCredentials
from agent_platform.integrations.reranking.cohere.config import CohereRerankerConfig
from agent_platform.integrations.reranking.langchain_base import LangChainReranker


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
            # CohereRerank defaults top_n=3, which silently truncates
            # results before our own top_k slice ever runs.
            top_n=config.top_k,
            cohere_api_key=self._credentials.api_key,
        )
