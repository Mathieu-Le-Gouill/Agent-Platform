from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from agent_platform.core.config import ProviderConfig
from agent_platform.core.schemas.vector import SparseVector

ConfigT = TypeVar("ConfigT", bound=ProviderConfig)


class BaseSparseEmbeddingProvider(ABC, Generic[ConfigT]):
    """Produces a sparse (term-weighted) embedding of a single text.

    Local/offline compute only (e.g. BM25 term-frequency), so, like
    `BaseClassificationProvider`, there is no sync/async pairing: one async
    method is enough.
    """

    @abstractmethod
    async def embed_sparse(
        self,
        text: str,
        config: ConfigT | None = None,
    ) -> SparseVector: ...
