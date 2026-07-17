from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

pytest.importorskip("boto3")

from agent_platform.integrations.ocr.aws_textract.aws_textract import (
    AWSTextractOCR,
    _from_textract,
)
from agent_platform.integrations.ocr.aws_textract.config import AWSTextractConfig
from agent_platform.integrations.credentials import AWSTextractCredentials


FAKE_TEXTRACT_RESPONSE = {
    "Blocks": [
        {"BlockType": "PAGE"},
        {"BlockType": "LINE", "Text": "Invoice #123", "Confidence": 98.0},
    ]
}


@pytest.mark.asyncio
async def test_extract_returns_chunks_from_line_blocks():

    document_id = uuid4()

    with patch(
        "agent_platform.integrations.ocr.aws_textract.aws_textract.boto3.client"
    ) as mock_boto:
        mock_client = MagicMock()
        mock_client.detect_document_text.return_value = FAKE_TEXTRACT_RESPONSE
        mock_boto.return_value = mock_client

        ocr = AWSTextractOCR(AWSTextractCredentials())

        with patch(
            "agent_platform.integrations.ocr.aws_textract.aws_textract.load_bytes",
            return_value=b"fake-bytes",
        ):
            chunks = await ocr.extract("some/path.png", document_id=document_id)

    assert [c.text for c in chunks] == ["Invoice #123"]
    assert chunks[0].document_id == document_id
    mock_client.detect_document_text.assert_called_once_with(
        Document={"Bytes": b"fake-bytes"}
    )


def test_from_textract_only_keeps_line_blocks():
    document_id = uuid4()
    chunks = _from_textract(
        FAKE_TEXTRACT_RESPONSE, document_id=document_id, min_confidence=0.5
    )
    assert len(chunks) == 1
    assert chunks[0].text == "Invoice #123"
    assert chunks[0].document_id == document_id
