from __future__ import annotations
from typing import Protocol
from agent_platform.models.embeddable import Embeddable
from agent_platform.models.cluster import Cluster

    
class BaseClusterer(Protocol):

    async def clusterize(self, items: list[Embeddable]) -> list[Cluster]: 
        ...