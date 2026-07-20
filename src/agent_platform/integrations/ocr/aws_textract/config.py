from __future__ import annotations
from pydantic import Field
from agent_platform.core.interfaces.ocr.config import OCRConfig


class AWSTextractConfig(OCRConfig):
    # Inherited from OCRConfig but unused here: detect_document_text/
    # analyze_document take no language parameter — Textract's OCR is
    # language-agnostic. https://docs.aws.amazon.com/textract/latest/dg/API_DetectDocumentText.html
    language: str = Field(default="eng", exclude=True)
    # AWS region for the Textract client. https://docs.aws.amazon.com/general/latest/gr/textract.html
    region_name: str = "us-east-1"
