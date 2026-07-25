from __future__ import annotations

from agent_platform.utils.lazy_imports import make_lazy_provider_accessors

_PROVIDERS: dict[str, str] = {
    "TesseractOCR": "agent_platform.integrations.ocr.tesseract.provider",
    "GoogleVisionOCR": "agent_platform.integrations.ocr.google_vision.provider",
    "AWSTextractOCR": "agent_platform.integrations.ocr.aws_textract.provider",
    "MistralOCR": "agent_platform.integrations.ocr.mistral.provider",
}

__getattr__, __dir__ = make_lazy_provider_accessors(_PROVIDERS, globals())
