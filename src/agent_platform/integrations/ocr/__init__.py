from __future__ import annotations

from agent_platform.utils.lazy_imports import make_lazy_provider_accessors

_PROVIDERS: dict[str, str] = {
    "TesseractOCR": "agent_platform.integrations.ocr.tesseract.tesseract",
    "GoogleVisionOCR": "agent_platform.integrations.ocr.google_vision.google_vision",
    "AWSTextractOCR": "agent_platform.integrations.ocr.aws_textract.aws_textract",
    "MistralOCR": "agent_platform.integrations.ocr.mistral.mistral",
}

__getattr__, __dir__ = make_lazy_provider_accessors(_PROVIDERS, globals())
