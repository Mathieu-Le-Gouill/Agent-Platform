from __future__ import annotations
from typing import Protocol
from agent_platform.models.embeddable import Embeddable

    
class BaseClassifier(Protocol):

    async def classify(self, items: list[Embeddable]) -> list[str]: 
        ...