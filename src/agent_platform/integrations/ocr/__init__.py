from __future__ import annotations

from importlib import import_module
from typing import Any

_PROVIDERS: dict[str, str] = {
    "TesseractOCR": "agent_platform.integrations.ocr.tesseract.tesseract",
    "GoogleVisionOCR": "agent_platform.integrations.ocr.google_vision.google_vision",
    "AWSTextractOCR": "agent_platform.integrations.ocr.aws_textract.aws_textract",
    "MistralOCR": "agent_platform.integrations.ocr.mistral.mistral",
}


def __getattr__(name: str) -> Any:
    if name in _PROVIDERS:
        return getattr(import_module(_PROVIDERS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_PROVIDERS.keys()))
