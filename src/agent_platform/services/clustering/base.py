from __future__ import annotations
from typing import Protocol
from models.protocols.text_unit import TextUnit
from models.cluster import Cluster

    
class BaseClusterer(Protocol):

    async def clusterize(self, items: list[TextUnit]) -> list[Cluster]: 
        ...