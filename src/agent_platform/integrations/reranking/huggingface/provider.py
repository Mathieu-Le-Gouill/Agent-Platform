from __future__ import annotations

import asyncio
from collections.abc import Sequence

from sentence_transformers import CrossEncoder

from agent_platform.core.interfaces.reranking.base import BaseReranker
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.integrations.reranking.huggingface.config import (
    HuggingFaceRerankerConfig,
)
from agent_platform.integrations.reranking.scoring import apply_rerank_results


class HuggingFaceRerankerProvider(BaseReranker[TextChunk, HuggingFaceRerankerConfig]):
    """Local cross-encoder reranking via sentence-transformers directly (no
    LangChain intermediate). Exempt from the retry/error-translation wrapping
    `NativeReranker` applies to network-bound providers, same reasoning as
    `flashrank`: retrying a deterministic local failure wastes CPU/GPU time
    instead of recovering from it."""

    def _default_config(self) -> HuggingFaceRerankerConfig:
        return HuggingFaceRerankerConfig()

    def _client(self, config: HuggingFaceRerankerConfig) -> CrossEncoder:
        return CrossEncoder(config.model, device=config.device)

    def rerank(
        self,
        query: str,
        items: Sequence[TextChunk],
        config: HuggingFaceRerankerConfig | None = None,
    ) -> Sequence[TextChunk]:
        config = config or self._default_config()
        if not items:
            return []

        client = self._client(config)
        scores = client.predict([(query, item.text) for item in items])
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)

        return apply_rerank_results(
            items,
            ranked,
            config,
            index=lambda r: r[0],
            score=lambda r: float(r[1]),
        )

    async def arerank(
        self,
        query: str,
        items: Sequence[TextChunk],
        config: HuggingFaceRerankerConfig | None = None,
    ) -> Sequence[TextChunk]:
        return await asyncio.to_thread(self.rerank, query, items, config)
