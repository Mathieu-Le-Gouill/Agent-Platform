from langchain_community.document_compressors import JinaRerank

from agent_platform.integrations.credentials import JinaCredentials
from agent_platform.integrations.reranking.jina.config import JinaRerankerConfig
from agent_platform.integrations.reranking.langchain_base import LangChainReranker


class JinaRerankerProvider(LangChainReranker[JinaRerankerConfig]):
    def __init__(self, credentials: JinaCredentials | None = None) -> None:
        self._credentials = (
            credentials if credentials is not None else JinaCredentials()
        )

    def _default_config(self) -> JinaRerankerConfig:
        return JinaRerankerConfig()

    def _client(self, config: JinaRerankerConfig) -> JinaRerank:
        api_key = (
            self._credentials.api_key.get_secret_value()
            if self._credentials.api_key is not None
            else None
        )
        return JinaRerank(
            model=config.model,
            # JinaRerank defaults top_n=3, which silently truncates results
            # before our own top_k slice ever runs.
            top_n=config.top_k,
            jina_api_key=api_key,
        )
