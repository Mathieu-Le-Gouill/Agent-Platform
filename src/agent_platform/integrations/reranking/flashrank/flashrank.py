from __future__ import annotations

import asyncio
from typing import Sequence

from flashrank import Ranker, RerankRequest

from agent_platform.core.interfaces.reranking.base import BaseReranker
from agent_platform.integrations.reranking.flashrank.config import FlashRankConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.errors import ProviderError


class FlashRankReranker(BaseReranker[TextChunk, FlashRankConfig]):
    def __init__(self) -> None:
        self._ranker: Ranker | None = None

    def _default_config(self) -> FlashRankConfig:
        return FlashRankConfig()

    def _get_ranker(self, config: FlashRankConfig) -> Ranker:
        if self._ranker is None:
            kwargs = {"model_name": config.model}
            if config.cache_dir:
                kwargs["cache_dir"] = config.cache_dir
            self._ranker = Ranker(**kwargs)
        return self._ranker

    async def rerank(
        self,
        query: str,
        items: Sequence[TextChunk],
        config: FlashRankConfig | None = None,
    ) -> Sequence[TextChunk]:
        config = config or self._default_config()
        if not items:
            return []

        ranker = await asyncio.to_thread(self._get_ranker, config)
        passages = [{"id": i, "text": item.text} for i, item in enumerate(items)]
        request = RerankRequest(query=query, passages=passages)

        try:
            results = await asyncio.to_thread(ranker.rerank, request)
        except Exception as exc:
            raise ProviderError(f"FlashRank reranking failed: {exc}") from exc

        reranked = [items[r["id"]] for r in results]

        if config.top_k is not None:
            reranked = reranked[: config.top_k]

        return reranked
