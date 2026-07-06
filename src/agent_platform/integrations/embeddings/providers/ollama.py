from langchain_ollama import OllamaEmbeddings

from agent_platform.integrations.embeddings.langchain_base import LangChainEmbedder
from agent_platform.integrations.embeddings.config import OllamaEmbeddingConfig


class OllamaEmbeddingProvider(LangChainEmbedder):
    def __init__(
        self,
        model="nomic-embed-text",
        config: OllamaEmbeddingConfig | None = None,
    ):

        self._model = model
        self.config = config or OllamaEmbeddingConfig()

        super().__init__()

    def _build_client(self):

        return OllamaEmbeddings(
            model=self._model,
            base_url=self.config.base_url,
        )
