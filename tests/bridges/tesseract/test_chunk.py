from uuid import uuid4

import pytest

from agent_platform.bridges.tesseract.chunk import from_tesseract


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

    chunks = from_tesseract(data, document_id=document_id, min_confidence=0.5)

    assert [c.text for c in chunks] == ["Hello", "world"]
    assert all(c.document_id == document_id for c in chunks)
    assert [c.index for c in chunks] == [0, 1]
    assert chunks[0].metadata.extra["confidence"] == pytest.approx(0.95)
    assert chunks[0].metadata.extra["bbox"]["left"] == 0