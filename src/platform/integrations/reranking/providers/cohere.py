from langchain_cohere import CohereRerank

from integrations.reranking.langchain_base import LangChainReranker

from integrations.reranking.config import CohereRerankerConfig


class CohereRerankerProvider(LangChainReranker):

    def __init__(
        self,
        model="rerank-english-v3.0",
        config: CohereRerankerConfig | None = None,
    ):
        self._model = model
        self.config = config or CohereRerankerConfig()

        super().__init__()


    def _client(self) -> CohereRerank:

        return CohereRerank(
            model=self._model,
            cohere_api_key=self.config.api_key,
        )