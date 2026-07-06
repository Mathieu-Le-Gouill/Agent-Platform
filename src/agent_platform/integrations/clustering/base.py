from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from agent_platform.models.chunk import TextChunk

from .config import ClusteringConfig
from .response import ClusterResult

ClusteringConfigT = TypeVar("ClusteringConfigT", bound=ClusteringConfig)


class ClusteringAlgorithm(ABC, Generic[ClusteringConfigT]):
    @abstractmethod
    async def clusterize(
        self,
        items: list[TextChunk],
        config: ClusteringConfigT | None = None,
    ) -> ClusterResult: ...
