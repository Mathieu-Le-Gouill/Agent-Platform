from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class OCREngine(str, Enum):
    TESSERACT = "tesseract"
    GOOGLE_VISION = "google_vision"
    AWS_TEXTRACT = "aws_textract"


class OCRConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, frozen=True)
    language: str = "eng"
    min_confidence: float = 0.0
