from unittest.mock import patch
from uuid import uuid4

from agent_platform.integrations.ocr.providers.tesseract import _from_tesseract
from agent_platform.models.chunk import TextChunk


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
