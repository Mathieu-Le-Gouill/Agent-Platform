from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from agent_platform.core.interfaces.classification.config import ClassificationConfig
from agent_platform.core.interfaces.classification.response import ClassificationResult

ConfigT = TypeVar("ConfigT", bound=ClassificationConfig)


class BaseClassificationProvider(ABC, Generic[ConfigT]):
    @abstractmethod
    async def classify(
        self,
        text: str,
        candidate_labels: list[str],
        config: ConfigT | None = None,
    ) -> ClassificationResult: ...
