from __future__ import annotations

from typing import Generic, TypeVar
from uuid import uuid4

from agent_platform.components.base import Component
from agent_platform.core.interfaces.ocr.base import BaseOCR
from agent_platform.core.interfaces.ocr.config import OCRConfig
from agent_platform.core.schemas.chunk import TextChunk

OCRConfigT = TypeVar("OCRConfigT", bound=OCRConfig)

OCRInput = tuple[str, OCRConfigT | None]


class OCR(Component[OCRInput[OCRConfigT], list[TextChunk]], Generic[OCRConfigT]):
    def __init__(self, backend: BaseOCR[OCRConfigT]) -> None:
        self._backend = backend

    async def arun(self, input: OCRInput[OCRConfigT]) -> list[TextChunk]:
        source, config = input
        results = await self._backend.extract(
            source=source, config=config, document_id=uuid4()
        )
        return list(results)
