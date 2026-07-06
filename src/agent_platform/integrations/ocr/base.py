from abc import ABC, abstractmethod
from typing import Generic, Sequence, TypeVar
from uuid import UUID

from agent_platform.integrations.ocr.config import OCRConfig
from agent_platform.models.chunk import TextChunk

OCRConfigT = TypeVar("OCRConfigT", bound=OCRConfig)


class BaseOCR(ABC, Generic[OCRConfigT]):
    @abstractmethod
    async def extract(
        self,
        source: str,
        config: OCRConfigT | None = None,
        document_id: UUID | None = None,
    ) -> Sequence[TextChunk]: ...
