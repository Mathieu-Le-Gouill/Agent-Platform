from uuid import UUID

import pytest

from agent_platform.components.ocr import OCR
from agent_platform.core.interfaces.ocr.base import BaseOCR
from agent_platform.core.interfaces.ocr.config import OCRConfig
from agent_platform.core.schemas.chunk import TextChunk


class _FakeOCR(BaseOCR):
    def __init__(self) -> None:
        self.calls: list[tuple[str, OCRConfig | None, UUID | None]] = []

    async def extract(self, source, config=None, document_id=None):
        self.calls.append((source, config, document_id))
        return [TextChunk(text="extracted", index=0)]


@pytest.mark.asyncio
async def test_forwards_source_and_config_to_backend():
    backend = _FakeOCR()
    ocr = OCR(backend)
    config = OCRConfig(language="fra")

    result = await ocr.arun(("img.png", config))

    assert result[0].text == "extracted"
    assert backend.calls[0][0] == "img.png"
    assert backend.calls[0][1] is config


@pytest.mark.asyncio
async def test_generates_a_document_id_when_none_given():
    backend = _FakeOCR()
    ocr = OCR(backend)

    await ocr.arun(("img.png", None))

    assert isinstance(backend.calls[0][2], UUID)
