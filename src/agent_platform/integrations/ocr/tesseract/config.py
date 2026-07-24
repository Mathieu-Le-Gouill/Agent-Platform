from __future__ import annotations

from agent_platform.core.interfaces.ocr.config import OCRConfig


class TesseractConfig(OCRConfig):
    # Tesseract 3-letter language code(s), e.g. "eng" or "eng+fra".
    # https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html
    language: str = "eng"
    # Page segmentation mode; 3 = fully automatic, no OSD.
    # https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html#page-segmentation-method
    psm: int = 3
    # OCR engine mode; 3 = default (LSTM + legacy, whichever is available).
    # https://github.com/tesseract-ocr/tesseract/blob/main/doc/tesseract.1.asc
    oem: int = 3
    # Path to a non-default tesseract binary. https://pypi.org/project/pytesseract
    tesseract_cmd: str | None = None
