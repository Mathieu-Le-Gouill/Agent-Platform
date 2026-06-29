from langchain_community.document_compressors import JinaRerank

from integrations.reranking.langchain_base import LangChainReranker

from integrations.reranking.config import JinaRerankerConfig


class JinaRerankerProvider(LangChainReranker):

    def __init__(
        self,
        model="jina-reranker-v2-base-multilingual",
        config=None,
    ):
        self._model = model
        self.config = (
            config
            or JinaRerankerConfig()
        )

        super().__init__()


    def _client(self) -> JinaRerank:

        api_key = (
            self.config.api_key.get_secret_value()
            if self.config.api_key is not None
            else None
        )

        return JinaRerank(
            model=self._model,
            jina_api_key=api_key,
        )