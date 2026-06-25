from __future__ import annotations
from typing import Protocol
from models.protocols.text_unit import TextUnit

    
class BaseClassifier(Protocol):

    async def classify(self, items: list[TextUnit]) -> list[str]: 
        ...