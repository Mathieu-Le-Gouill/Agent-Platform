from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Generic, TypeVar

from agent_platform.core.schemas.chunk import TextChunk

from agent_platform.core.interfaces.clustering.config import ClusteringConfig
from agent_platform.core.interfaces.clustering.response import ClusterResult

ClusteringConfigT = TypeVar(
    "ClusteringConfigT", bound=ClusteringConfig, contravariant=True
)


class BaseClusteringAlgorithm(ABC, Generic[ClusteringConfigT]):
    @abstractmethod
    async def clusterize(
        self,
        items: list[TextChunk],
        config: ClusteringConfigT | None = None,
    ) -> ClusterResult: ...
