from __future__ import annotations
from typing import Protocol
from core.value_objects.embeddable import Embeddable
from core.entities.cluster import Cluster

    
class ClustererPort(Protocol):

    async def clusterize(self, items: list[Embeddable]) -> list[Cluster]: 
        ...