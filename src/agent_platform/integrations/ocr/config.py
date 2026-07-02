from dataclasses import dataclass
from enum import Enum


class OCREngine(str, Enum):
    TESSERACT = "tesseract"
    GOOGLE_VISION = "google_vision"
    AWS_TEXTRACT = "aws_textract"


@dataclass(slots=True, frozen=True)
class OCRConfig:
    language: str = "eng"
    min_confidence: float = 0.0


@dataclass(slots=True, frozen=True)
class TesseractConfig(OCRConfig):
    tesseract_cmd: str | None = None
    psm: int = 3
    oem: int = 3


@dataclass(slots=True, frozen=True)
class GoogleVisionConfig(OCRConfig):
    credentials_path: str | None = None
    feature_type: str = "DOCUMENT_TEXT_DETECTION"


@dataclass(slots=True, frozen=True)
class AWSTextractConfig(OCRConfig):
    region_name: str = "us-east-1"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None