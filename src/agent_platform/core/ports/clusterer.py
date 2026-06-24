from typing import Protocol
from core.entities.cluster import Cluster
from platform.core.ports.embeddable import Embeddable

    
class ClustererPort(Protocol):

    async def clusterize(self, items: list[Embeddable]) -> list[Cluster]: 
        ...