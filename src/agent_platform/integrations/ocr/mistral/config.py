from __future__ import annotations
from typing import Literal

from pydantic import Field
from agent_platform.core.interfaces.ocr.config import OCRConfig


class MistralOCRConfig(OCRConfig):
    # Inherited from OCRConfig but unused here: Mistral OCR is multilingual by
    # default and takes no language parameter. https://docs.mistral.ai/api/endpoint/ocr
    language: str = Field(default="eng", exclude=True)
    # Model id for the OCR endpoint. https://docs.mistral.ai/api/endpoint/ocr
    model: str = "mistral-ocr-latest"
    # Requests per-word or per-page confidence scores so min_confidence can
    # actually filter results; None omits scores entirely (API default).
    # https://docs.mistral.ai/api/endpoint/ocr
    confidence_scores_granularity: Literal["word", "page"] | None = None
    # Restrict OCR to specific 0-indexed pages, or a page-range string.
    # https://docs.mistral.ai/api/endpoint/ocr
    pages: list[int] | str | None = None
    # Output format for extracted tables. https://docs.mistral.ai/api/endpoint/ocr
    table_format: Literal["markdown", "html"] | None = None
