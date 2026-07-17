from __future__ import annotations
from agent_platform.core.interfaces.ocr.config import OCRConfig


class AWSTextractConfig(OCRConfig):
    region_name: str = "us-east-1"
