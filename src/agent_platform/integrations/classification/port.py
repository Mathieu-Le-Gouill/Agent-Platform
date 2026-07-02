from __future__ import annotations
from abc import ABC, abstractmethod
from agent_platform.models.protocols.text_unit import TextUnit

    
class ClassificationModel(ABC):

    @abstractmethod
    async def classify(self, items: list[TextUnit]) -> list[str]: 
        ...