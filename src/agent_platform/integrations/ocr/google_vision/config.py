from __future__ import annotations

from agent_platform.core.interfaces.ocr.config import OCRConfig


class GoogleVisionConfig(OCRConfig):
    # BCP-47 language hints (e.g. "en", not Tesseract-style "eng"), passed as
    # ImageContext.language_hints. https://cloud.google.com/vision/docs/languages
    # https://cloud.google.com/vision/docs/reference/rest/v1/ImageContext
    language_hints: list[str] = []
    # "TEXT_DETECTION" (sparse text, e.g. signage) or "DOCUMENT_TEXT_DETECTION"
    # (dense text, e.g. scanned pages) — selects the client method called.
    # https://cloud.google.com/vision/docs/ocr
    feature_type: str = "DOCUMENT_TEXT_DETECTION"
