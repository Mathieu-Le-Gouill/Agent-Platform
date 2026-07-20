from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.embeddings import Embeddings

from agent_platform.core.interfaces.vector_store.base import BaseVectorStore
from agent_platform.core.interfaces.vector_store.config import DistanceMetric
from agent_platform.integrations.vector_store.faiss.config import FAISSConfig
from agent_platform.integrations.vector_store.langchain_base import (
    _chunk_to_lc,
    _lc_to_chunk,
)
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import Score
from agent_platform.core.errors import ProviderError, error_logged, with_retry

# DistanceMetric has no JACCARD/MAX_INNER_PRODUCT/DOT counterpart, so those
# LangChain strategies are unreachable via config; COSINE/EUCLIDEAN/DOT map 1:1.
# Source: api.python.langchain.com langchain_community.vectorstores.faiss.DistanceStrategy.
_DISTANCE_STRATEGY_MAP: dict[DistanceMetric, DistanceStrategy] = {
    DistanceMetric.COSINE: DistanceStrategy.COSINE,
    DistanceMetric.EUCLIDEAN: DistanceStrategy.EUCLIDEAN_DISTANCE,
    DistanceMetric.DOT: DistanceStrategy.DOT_PRODUCT,
}


class FAISSStore(BaseVectorStore[FAISSConfig]):
    def __init__(
        self,
        embeddings: Embeddings | None = None,
    ) -> None:
        self._embeddings = embeddings
        self._store: FAISS | None = None

    def _default_config(self) -> FAISSConfig:
        return FAISSConfig()

    def _load_or_none(self, config: FAISSConfig) -> FAISS | None:
        if self._store is not None:
            return self._store
        if config.index_path and Path(config.index_path).exists():
            self._store = FAISS.load_local(
                config.index_path,
                embeddings=self._embeddings,
                allow_dangerous_deserialization=True,
                distance_strategy=_DISTANCE_STRATEGY_MAP[config.distance],
            )
        return self._store

    async def add(
        self, documents: list[TextChunk], config: FAISSConfig | None = None
    ) -> None:
        config = config or self._default_config()
        lc_docs = [_chunk_to_lc(doc) for doc in documents]

        if self._store is None:
            if self._embeddings is None:
                raise ProviderError(
                    "Embeddings are required to initialize a FAISS index"
                )
            self._store = await asyncio.to_thread(
                FAISS.from_documents,
                lc_docs,
                self._embeddings,
                distance_strategy=_DISTANCE_STRATEGY_MAP[config.distance],
            )
        else:
            await self._store.aadd_documents(lc_docs)

        if config.index_path:
            await asyncio.to_thread(self._store.save_local, config.index_path)

    async def delete(
        self, document_ids: list[UUID], config: FAISSConfig | None = None
    ) -> None:
        config = config or self._default_config()
        store = self._load_or_none(config)
        if store is None:
            return
        ids = [str(doc_id) for doc_id in document_ids]
        await asyncio.to_thread(store.delete, ids)

    @error_logged(re_raise=ProviderError, message="Vector store search failed")
    @with_retry()
    async def search(
        self,
        query_vector: list[float],
        k: int = 5,
        config: FAISSConfig | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        config = config or self._default_config()
        store = self._load_or_none(config)
        if store is None:
            return []
        results = await asyncio.to_thread(
            store.similarity_search_by_vector, query_vector, k, filter=filter
        )
        return [_lc_to_chunk(r) for r in results]

    @error_logged(re_raise=ProviderError, message="Vector store search failed")
    @with_retry()
    async def search_with_scores(
        self,
        query_vector: list[float],
        k: int = 5,
        config: FAISSConfig | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[TextChunk, Score]]:
        config = config or self._default_config()
        store = self._load_or_none(config)
        if store is None:
            return []
        results = await asyncio.to_thread(
            store.similarity_search_by_vector_with_relevance_scores,
            query_vector,
            k,
            filter=filter,
        )
        # Clamped to [0, 1]; now correctly corresponds to the configured
        # distance_strategy rather than always assuming the library default.
        return [
            (_lc_to_chunk(doc), Score.similarity(min(1.0, max(0.0, float(score)))))
            for doc, score in results
        ]
