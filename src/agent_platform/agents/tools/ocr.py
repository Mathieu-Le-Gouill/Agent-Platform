from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, Field

from agent_platform.agents.tools.base import Tool
from agent_platform.agents.tools._utils import safe_call
from agent_platform.core.interfaces.ocr.base import BaseOCR
from agent_platform.core.interfaces.ocr.config import OCRConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.utils.score import filter_by_score


class OCRInput(BaseModel):
    source: str = Field(..., description="Image file path or URL")
    language: str = Field(
        default="eng",
        min_length=2,
        max_length=10,
        description="Language hint (ISO 639-1 code or Tesseract format)",
    )
    min_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold [0-1]",
    )


class OCRTool(Tool):
    name = "ocr"
    description = "Extract text from an image file or URL using OCR."
    input_schema = OCRInput
    output_schema = None

    def __init__(self, provider: BaseOCR) -> None:
        self._provider = provider

    async def run(self, **kwargs) -> list[TextChunk]:
        validated = OCRInput(**kwargs)
        config = OCRConfig(
            language=validated.language,
            min_confidence=validated.min_confidence,
        )
        results = await safe_call(
            self._provider.extract(
                source=validated.source,
                config=config,
                document_id=uuid4(),
            ),
            "OCR extraction failed",
        )
        return filter_by_score(
            results,
            validated.min_confidence,
            key=lambda c: c.metadata.get("confidence") or 1.0,
        )
