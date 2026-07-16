from enum import Enum

from agent_platform.core.schemas.config import ProviderConfig


class OCREngine(str, Enum):
    TESSERACT = "tesseract"
    GOOGLE_VISION = "google_vision"
    AWS_TEXTRACT = "aws_textract"


class OCRConfig(ProviderConfig):
    language: str = "eng"
    min_confidence: float = 0.0
