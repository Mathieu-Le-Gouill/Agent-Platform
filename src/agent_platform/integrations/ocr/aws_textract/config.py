from __future__ import annotations

from agent_platform.core.interfaces.ocr.config import OCRConfig


class AWSTextractConfig(OCRConfig):
    # AWS region for the Textract client. https://docs.aws.amazon.com/general/latest/gr/textract.html
    region_name: str = "us-east-1"
