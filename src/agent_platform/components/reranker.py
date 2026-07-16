from __future__ import annotations

from typing import Generic, TypeVar

from agent_platform.core.credentials import BaseCredentials
from agent_platform.core.interfaces.reranking.base import BaseReranker
from agent_platform.core.interfaces.reranking.config import RerankerConfig
from agent_platform.components.base import Component

CredentialsT = TypeVar("CredentialsT", bound=BaseCredentials)
RerankerConfigT = TypeVar("RerankerConfigT", bound=RerankerConfig)

T = TypeVar("T")


class Reranker(Component[list[T], list[T]], Generic[CredentialsT, T, RerankerConfigT]):
    def __init__(
        self,
        backend: BaseReranker[CredentialsT, T, RerankerConfigT],
        config: RerankerConfigT | None = None,
    ) -> None:
        self._backend = backend
        self._config = config

    async def arun(self, input: list[T]) -> list[T]:
        return self._backend.rerank(input, self._config)
