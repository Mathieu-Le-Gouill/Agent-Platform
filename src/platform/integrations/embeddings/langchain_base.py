from abc import abstractmethod
from typing import Sequence

from langchain_core.embeddings import Embeddings

from integrations.embeddings.response import EmbeddingResponse
from integrations.embeddings.base import BaseEmbeddingProvider
from models.embedding import Embedding
from models.protocols.text_unit import TextUnit
from models.token import TokenUsage


class LangChainEmbedder(BaseEmbeddingProvider):
    _client: Embeddings
    _model: str


    def __init__(self):
        self._client = self._build_client()


    @abstractmethod
    def _build_client(self) -> Embeddings:
        ...


    def _extract_usage(self, raw: list[list[float]]) -> TokenUsage:
        return TokenUsage.zero()


    async def encode(
        self,
        items: Sequence[TextUnit],
    ) -> EmbeddingResponse:
        
        texts = [item.text for item in items]

        vectors: list[list[float]] = await self._client.aembed_documents(texts)

        embeddings = [
            Embedding.from_list(vector, model=self._model, id=item.id)
            for item, vector in zip(items, vectors)
        ]

        return EmbeddingResponse(
            embeddings=embeddings,
            model=self._model,
            usage=self._extract_usage(vectors),
        )