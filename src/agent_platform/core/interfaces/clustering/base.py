from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Generic, TypeVar

from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.credentials import BaseCredentials

from agent_platform.core.interfaces.clustering.config import ClusteringConfig
from agent_platform.core.interfaces.clustering.response import ClusterResult

ClusteringConfigT = TypeVar(
    "ClusteringConfigT", bound=ClusteringConfig, contravariant=True
)
CredentialsT = TypeVar("CredentialsT", bound=BaseCredentials, covariant=True)


class ClusteringAlgorithm(ABC, Generic[CredentialsT, ClusteringConfigT]):
    def __init__(self, credentials: CredentialsT) -> None:
        self._credentials = credentials

    @abstractmethod
    async def clusterize(
        self,
        items: list[TextChunk],
        config: ClusteringConfigT | None = None,
    ) -> ClusterResult: ...
