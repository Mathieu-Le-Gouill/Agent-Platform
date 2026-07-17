from __future__ import annotations
from pydantic import Field
from agent_platform.core.interfaces.ocr.config import OCRConfig


class GoogleVisionConfig(OCRConfig):
    language: str = "eng"
    feature_type: str = "DOCUMENT_TEXT_DETECTION"
