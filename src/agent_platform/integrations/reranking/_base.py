from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar, cast

from agent_platform.core.errors import ProviderError, error_logged, with_retry
from agent_platform.core.interfaces.reranking.base import BaseReranker, RerankerConfigT
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.integrations.reranking.scoring import apply_rerank_results

# The async client (e.g. `AsyncClient`), used by `arerank`/`_invoke_async`.
AsyncClientT = TypeVar("AsyncClientT")
# The sync client (e.g. `Client`), used by `rerank`/`_invoke_sync`. Equal to
# `AsyncClientT` for vendors with a single client class serving both.
SyncClientT = TypeVar("SyncClientT")
# One raw result per reranked document, already in the vendor's relevance
# order (e.g. a cohere `RerankResponseResultsItem`, a jina result dict).
ResultT = TypeVar("ResultT")


class NativeReranker(
    BaseReranker[TextChunk, RerankerConfigT],
    Generic[RerankerConfigT, AsyncClientT, SyncClientT, ResultT],
):
    """Shared request/response plumbing for network-bound reranking providers
    calling a vendor SDK or REST endpoint directly. Every subclass owns its
    own vendor-specific pieces (client construction, request assembly,
    result field access); this base owns the empty-input short-circuit,
    retry/error-translation, and result -> `TextChunk` mapping that are
    otherwise identical across providers. Local/offline rerankers (flashrank,
    huggingface) don't use this base: they have no network call to retry and
    no separate sync/async client to construct.
    """

    @abstractmethod
    def _default_config(self) -> RerankerConfigT: ...

    @abstractmethod
    def _async_client(self, config: RerankerConfigT) -> AsyncClientT: ...

    def _sync_client(self, config: RerankerConfigT) -> SyncClientT:
        # Default for vendors with one client class serving both sync and
        # async calls; overridden where the SDK splits them into distinct
        # classes (e.g. `Client`/`AsyncClient`).
        return cast(SyncClientT, self._async_client(config))

    @abstractmethod
    def _invoke_sync(
        self,
        client: SyncClientT,
        query: str,
        documents: list[str],
        config: RerankerConfigT,
    ) -> Sequence[ResultT]:
        """Assemble the native request and make the blocking call, returning
        results already sorted in relevance order."""

    @abstractmethod
    async def _invoke_async(
        self,
        client: AsyncClientT,
        query: str,
        documents: list[str],
        config: RerankerConfigT,
    ) -> Sequence[ResultT]:
        """Assemble the native request and make the async call, returning
        results already sorted in relevance order."""

    @abstractmethod
    def _result_index(self, result: ResultT) -> int:
        """Position of this result's document in the original `documents` list."""

    @abstractmethod
    def _result_score(self, result: ResultT) -> float | None: ...

    def rerank(
        self,
        query: str,
        items: Sequence[TextChunk],
        config: RerankerConfigT | None = None,
    ) -> Sequence[TextChunk]:
        config = config or self._default_config()
        if not items:
            return []

        client = self._sync_client(config)
        results = self._invoke_sync(
            client, query, [item.text for item in items], config
        )
        return apply_rerank_results(
            items,
            results,
            config,
            index=self._result_index,
            score=self._result_score,
        )

    @error_logged(re_raise=ProviderError, message="Reranking failed")
    @with_retry()
    async def arerank(
        self,
        query: str,
        items: Sequence[TextChunk],
        config: RerankerConfigT | None = None,
    ) -> Sequence[TextChunk]:
        config = config or self._default_config()
        if not items:
            return []

        client = self._async_client(config)
        results = await self._invoke_async(
            client, query, [item.text for item in items], config
        )
        return apply_rerank_results(
            items,
            results,
            config,
            index=self._result_index,
            score=self._result_score,
        )
