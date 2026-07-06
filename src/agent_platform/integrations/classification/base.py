from __future__ import annotations

from abc import ABC, abstractmethod

from agent_platform.models.chunk import TextChunk


class ClassificationModel(ABC):
    @abstractmethod
    async def classify(
        self,
        items: list[TextChunk],
        candidate_labels: list[str] | None = None,
    ) -> list[str]: ...
