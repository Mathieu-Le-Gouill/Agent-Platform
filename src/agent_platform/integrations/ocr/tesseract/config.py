from __future__ import annotations
from agent_platform.core.interfaces.ocr.config import OCRConfig


class TesseractConfig(OCRConfig):
    language: str = "eng"
    psm: int = 3
    oem: int = 3
    tesseract_cmd: str | None = None
