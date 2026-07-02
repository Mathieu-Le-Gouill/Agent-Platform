from langchain_cohere import CohereRerank

from agent_platform.integrations.reranking.langchain_base import LangChainReranker

from agent_platform.integrations.reranking.config import CohereRerankerConfig


class CohereRerankerProvider(LangChainReranker):

    def __init__(
        self,
        model="rerank-english-v3.0",
        config: CohereRerankerConfig = CohereRerankerConfig(),
    ):
        self._model = model
        self.config = config or CohereRerankerConfig()

        super().__init__()


    def _client(self) -> CohereRerank:

        return CohereRerank(
            model=self._model,
            cohere_api_key=self.config.api_key,
        )