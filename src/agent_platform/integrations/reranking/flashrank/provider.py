from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from flashrank import Ranker, RerankRequest

from agent_platform.core.errors import ProviderError
from agent_platform.core.interfaces.reranking.base import BaseReranker
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.integrations.reranking.flashrank.config import FlashRankConfig
from agent_platform.integrations.reranking.scoring import apply_rerank_results


class FlashRankReranker(BaseReranker[TextChunk, FlashRankConfig]):
    """Local reranking via flashrank. Exempt from the retry/error-translation
    wrapping `NativeReranker` applies to network-bound providers: retrying a
    deterministic local failure wastes CPU time instead of recovering from it.
    """

    def __init__(self) -> None:
        self._ranker: Ranker | None = None

    def _default_config(self) -> FlashRankConfig:
        return FlashRankConfig()

    def _get_ranker(self, config: FlashRankConfig) -> Ranker:
        if self._ranker is None:
            kwargs: dict[str, Any] = {"model_name": config.model}
            if config.cache_dir:
                kwargs["cache_dir"] = config.cache_dir
            if config.max_length is not None:
                kwargs["max_length"] = config.max_length
            self._ranker = Ranker(**kwargs)
        return self._ranker

    def rerank(
        self,
        query: str,
        items: Sequence[TextChunk],
        config: FlashRankConfig | None = None,
    ) -> Sequence[TextChunk]:
        config = config or self._default_config()
        if not items:
            return []

        ranker = self._get_ranker(config)
        passages = [{"id": i, "text": item.text} for i, item in enumerate(items)]
        request = RerankRequest(query=query, passages=passages)

        try:
            results = ranker.rerank(request)
        except Exception as exc:
            raise ProviderError(f"FlashRank reranking failed: {exc}") from exc

        return apply_rerank_results(
            items,
            results,
            config,
            index=lambda r: r["id"],
            score=lambda r: r.get("score"),
        )

    async def arerank(
        self,
        query: str,
        items: Sequence[TextChunk],
        config: FlashRankConfig | None = None,
    ) -> Sequence[TextChunk]:
        return await asyncio.to_thread(self.rerank, query, items, config)
