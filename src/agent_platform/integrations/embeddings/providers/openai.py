from langchain_openai import OpenAIEmbeddings

from agent_platform.integrations.embeddings.langchain_base import LangChainEmbedder
from agent_platform.integrations.embeddings.config import OpenAIEmbeddingConfig


class OpenAIEmbeddingProvider(LangChainEmbedder):
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        config: OpenAIEmbeddingConfig | None = None,
    ):
        self._model = model
        self.config = config or OpenAIEmbeddingConfig()

        super().__init__()

    def _build_client(self):

        return OpenAIEmbeddings(
            model=self._model,
            dimensions=self.config.dimensions,
            organization=self.config.organization,
            timeout=self.config.timeout,
        )
