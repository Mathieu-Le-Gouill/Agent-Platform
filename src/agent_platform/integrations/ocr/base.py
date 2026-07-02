from abc import ABC, abstractmethod
from typing import Sequence
from uuid import UUID

from agent_platform.integrations.ocr.config import OCRConfig
from agent_platform.models.chunk import Chunk

class BaseOCR(ABC):

    @abstractmethod
    async def extract(
        self,
        source: str,
        config: OCRConfig | None = None,
        document_id: UUID | None = None,
    ) -> Sequence[Chunk]:
        ...