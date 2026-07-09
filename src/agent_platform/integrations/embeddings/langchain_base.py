from __future__ import annotations

from abc import abstractmethod
from typing import Sequence, Generic

from langchain_core.embeddings import Embeddings

from agent_platform.integrations.embeddings.response import EmbeddingResponse
from agent_platform.integrations.embeddings.base import (
    BaseEmbeddingProvider,
    EmbeddingConfigT,
    CredentialsT,
)
from agent_platform.models.embedding import Embedding
from agent_platform.models.chunk import TextChunk


class LangChainEmbedder(BaseEmbeddingProvider[CredentialsT, EmbeddingConfigT], Generic[CredentialsT, EmbeddingConfigT]):

    @abstractmethod
    def _client(self, config: EmbeddingConfigT) -> Embeddings: ...
    
    @abstractmethod
    def _default_config(self) -> EmbeddingConfigT: ...

    def embed_document(
        self,
        items: Sequence[TextChunk],
        config: EmbeddingConfigT | None = None,
    ) -> EmbeddingResponse:
        
        config = config or self._default_config()
        lc = self._client(config)

        texts = [item.text for item in items]
        vectors: list[list[float]] = lc.embed_documents(texts)

        embeddings = [
            Embedding.from_list(vector, model=config.model, id=item.id)
            for item, vector in zip(items, vectors)
        ]

        return EmbeddingResponse(
            embeddings=embeddings,
            model=config.model,
        )


    async def aembed_document(
        self,
        items: Sequence[TextChunk],
        config: EmbeddingConfigT | None = None,
    ) -> EmbeddingResponse:
        config = config or self._default_config()
        lc = self._client(config)

        texts = [item.text for item in items]
        vectors: list[list[float]] = await lc.aembed_documents(texts)

        embeddings = [
            Embedding.from_list(vector, model=config.model, id=item.id)
            for item, vector in zip(items, vectors)
        ]

        return EmbeddingResponse(
            embeddings=embeddings,
            model=config.model,
        )
    

    def embed_query(
        self,
        query: str,
        config: EmbeddingConfigT | None = None,
    ) -> EmbeddingResponse:
        config = config or self._default_config()
        lc = self._client(config)

        vector: list[float] = lc.embed_query(query)

        embedding = Embedding.from_list(vector)

        return EmbeddingResponse(
            embeddings=[embedding],
            model=config.model,
        )


    async def aembed_query(
        self,
        query: str,
        config: EmbeddingConfigT | None = None,
    ) -> EmbeddingResponse:
        
        config = config or self._default_config()
        lc = self._client(config)

        vector: list[float] = await lc.aembed_query(query)

        embedding = Embedding.from_list(vector)

        return EmbeddingResponse(
            embeddings=[embedding],
            model=config.model,
        )