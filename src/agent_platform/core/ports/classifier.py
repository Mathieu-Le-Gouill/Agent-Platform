from __future__ import annotations
from typing import Protocol
from core.value_objects.embeddable import Embeddable

    
class Classifier(Protocol):

    async def classify(self, items: list[Embeddable]) -> list[str]: 
        ...