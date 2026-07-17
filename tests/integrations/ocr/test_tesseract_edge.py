from unittest.mock import patch
from uuid import uuid4

import pytest

pytest.importorskip("pytesseract")

from agent_platform.integrations.ocr.tesseract.tesseract import _from_tesseract
from agent_platform.core.schemas.chunk import TextChunk


def test_from_tesseract_missing_block_num():
    data = {
        "text": ["Hello", "", "World"],
        "conf": [95, 0, 80],
        "left": [0, 0, 10],
        "top": [0, 0, 20],
        "width": [100, 0, 50],
        "height": [50, 0, 30],
        "page_num": [1, 1, 1],
    }
    doc_id = uuid4()
    result = _from_tesseract(data, document_id=doc_id, min_confidence=0.0)
    assert len(result) == 2
    assert result[0].text == "Hello"
    assert result[1].text == "World"
    assert result[0].metadata["block_num"] is None
    assert result[1].metadata["block_num"] is None


def test_from_tesseract_skips_blank_and_low_confidence():
    document_id = uuid4()

    data = {
        "text": ["Hello", "", "world", "junk"],
        "conf": ["95", "-1", "88", "10"],
        "left": [0, 0, 10, 20],
        "top": [0, 0, 5, 5],
        "width": [50, 0, 40, 30],
        "height": [12, 0, 12, 12],
        "page_num": [1, 1, 1, 1],
    }

    chunks = _from_tesseract(data, document_id=document_id, min_confidence=50.0)

    assert [c.text for c in chunks] == ["Hello", "world"]
    assert all(c.document_id == document_id for c in chunks)
    assert chunks[0].confidence.value == pytest.approx(95.0)
    assert chunks[0].bbox.x == 0
