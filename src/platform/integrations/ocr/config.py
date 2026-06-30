from dataclasses import dataclass
from enum import Enum


class OCRStrategy(str, Enum):
    FAST = "fast"
    HI_RES = "hi_res"
    OCR_ONLY = "ocr_only"


@dataclass(slots=True, frozen=True)
class OCRConfig:
    strategy: OCRStrategy = OCRStrategy.HI_RES
    language: str = "eng"
    batch_size: int = 32
    include_metadata: bool = True


@dataclass(slots=True, frozen=True)
class UnstructuredOCRConfig(OCRConfig):
    extract_images: bool = False