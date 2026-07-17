from __future__ import annotations
from agent_platform.core.interfaces.ocr.config import OCRConfig


class MistralOCRConfig(OCRConfig):
    model: str = "mistral-ocr-latest"
