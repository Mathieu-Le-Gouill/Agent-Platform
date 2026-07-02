from uuid import uuid4

import pytest

from agent_platform.bridges.aws.chunk import from_textract


def test_from_textract_only_keeps_line_blocks():

    document_id = uuid4()

    response = {
        "Blocks": [
            {"BlockType": "PAGE"},
            {
                "BlockType": "LINE",
                "Text": "Invoice #123",
                "Confidence": 99.2,
                "Geometry": {"BoundingBox": {"Left": 0.1, "Top": 0.1}},
            },
            {
                "BlockType": "LINE",
                "Text": "low confidence junk",
                "Confidence": 10.0,
            },
        ]
    }

    chunks = from_textract(response, document_id=document_id, min_confidence=0.5)

    assert len(chunks) == 1
    assert chunks[0].text == "Invoice #123"
    assert chunks[0].document_id == document_id
    assert chunks[0].metadata.extra["confidence"] == pytest.approx( 0.992)