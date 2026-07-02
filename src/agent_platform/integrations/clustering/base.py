from __future__ import annotations
from abc import ABC, abstractmethod
from agent_platform.models.protocols.text_unit import TextUnit
from agent_platform.models.cluster import Cluster

    
class ClusteringAlgorithm(ABC):

    @abstractmethod
    async def clusterize(self, items: list[TextUnit]) -> list[Cluster]: 
        ...