from __future__ import annotations

from typing import Generic, TypeVar

from agent_platform.core.credentials import BaseCredentials
from agent_platform.core.interfaces.reranking.base import BaseReranker
from agent_platform.core.interfaces.reranking.config import RerankerConfig
from agent_platform.components.base import Component
from agent_platform.utils.batching import chunked

CredentialsT = TypeVar("CredentialsT", bound=BaseCredentials)
RerankerConfigT = TypeVar("RerankerConfigT", bound=RerankerConfig)

T = TypeVar("T")

RerankerInput = tuple[str, list[T]]


class Reranker(
    Component[RerankerInput[T], list[T]], Generic[CredentialsT, T, RerankerConfigT]
):
    def __init__(
        self,
        backend: BaseReranker[CredentialsT, T, RerankerConfigT],
        config: RerankerConfigT | None = None,
    ) -> None:
        self._backend = backend
        self._config = config

    async def arun(self, input: RerankerInput[T]) -> list[T]:
        query, items = input
        batch_size = (self._config or RerankerConfig()).batch_size

        results: list[T] = []
        for batch in chunked(items, batch_size):
            reranked = await self._backend.rerank(query, list(batch), self._config)
            results.extend(reranked)

        return results
