from langchain_mistralai import MistralAIEmbeddings

from integrations.embeddings.langchain_base import LangChainEmbedder
from integrations.embeddings.config import MistralEmbeddingConfig


class MistralEmbeddingProvider(
    LangChainEmbedder
):

    def __init__(
        self,
        model="mistral-embed",
        config: MistralEmbeddingConfig | None = None,
    ):

        self._model = model
        self.config = config or MistralEmbeddingConfig()

        super().__init__()


    def _build_client(self):

        return MistralAIEmbeddings(
            model=self._model,
            timeout=self.config.timeout or 120,
        )