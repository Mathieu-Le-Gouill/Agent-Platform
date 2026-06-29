from langchain_community.embeddings import HuggingFaceEmbeddings

from integrations.embeddings.langchain_base import LangChainEmbedder
from integrations.embeddings.config import SentenceTransformerConfig


class SentenceTransformerEmbeddingProvider(
    LangChainEmbedder
):

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        config: SentenceTransformerConfig | None = None,
    ):
        self._model = model
        self.config = config or SentenceTransformerConfig()

        super().__init__()


    def _build_client(self):

        return HuggingFaceEmbeddings(
            model_name=self._model,
            model_kwargs=self.config.model_kwargs or {},
            encode_kwargs={
                "normalize_embeddings":
                    self.config.normalize,
                    **(self.config.encode_kwargs or {})
            },
        )