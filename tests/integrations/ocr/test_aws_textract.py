from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from agent_platform.integrations.ocr.providers.aws_textract import AWSTextractOCR
from agent_platform.integrations.ocr.config import AWSTextractConfig


FAKE_TEXTRACT_RESPONSE = {
    "Blocks": [
        {"BlockType": "PAGE"},
        {"BlockType": "LINE", "Text": "Invoice #123", "Confidence": 98.0},
    ]
}


@pytest.mark.asyncio
async def test_extract_returns_chunks_from_line_blocks():

    document_id = uuid4()

    with patch("integrations.ocr.aws_textract.boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_client.detect_document_text.return_value = FAKE_TEXTRACT_RESPONSE
        mock_boto.return_value = mock_client

        ocr = AWSTextractOCR(AWSTextractConfig())

        with patch.object(AWSTextractOCR, "_load_bytes", return_value=b"fake-bytes"):
            chunks = await ocr.extract("some/path.png", document_id=document_id)

    assert [c.text for c in chunks] == ["Invoice #123"]
    assert chunks[0].document_id == document_id
    mock_client.detect_document_text.assert_called_once_with(
        Document={"Bytes": b"fake-bytes"}
    )