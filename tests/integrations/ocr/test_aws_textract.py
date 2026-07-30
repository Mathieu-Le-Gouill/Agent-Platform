from unittest.mock import MagicMock
from uuid import uuid4

import pytest

pytest.importorskip("boto3")

from agent_platform.integrations.credentials import AWSTextractCredentials
from agent_platform.integrations.ocr.aws_textract.config import AWSTextractConfig
from agent_platform.integrations.ocr.aws_textract.mappers import (
    from_textract as _from_textract,
)
from agent_platform.integrations.ocr.aws_textract.provider import AWSTextractOCR

FAKE_TEXTRACT_RESPONSE = {
    "Blocks": [
        {"BlockType": "PAGE"},
        {"BlockType": "LINE", "Text": "Invoice #123", "Confidence": 98.0},
    ]
}


@pytest.mark.asyncio
async def test_extract_returns_chunks_from_line_blocks(mocker):

    document_id = uuid4()

    mock_boto = mocker.patch(
        "agent_platform.integrations.ocr.aws_textract.provider.boto3.client"
    )
    mock_client = MagicMock()
    mock_client.detect_document_text.return_value = FAKE_TEXTRACT_RESPONSE
    mock_boto.return_value = mock_client

    ocr = AWSTextractOCR(AWSTextractCredentials())

    mocker.patch(
        "agent_platform.integrations.ocr.aws_textract.provider.load_bytes",
        return_value=b"fake-bytes",
    )
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


@pytest.mark.asyncio
async def test_client_is_cached_across_calls_for_same_region(mocker):
    mock_boto = mocker.patch(
        "agent_platform.integrations.ocr.aws_textract.provider.boto3.client"
    )
    mock_client = MagicMock()
    mock_client.detect_document_text.return_value = FAKE_TEXTRACT_RESPONSE
    mock_boto.return_value = mock_client

    ocr = AWSTextractOCR(AWSTextractCredentials())

    mocker.patch(
        "agent_platform.integrations.ocr.aws_textract.provider.load_bytes",
        return_value=b"fake-bytes",
    )
    await ocr.extract("a.png")
    await ocr.extract("b.png")

    mock_boto.assert_called_once()


@pytest.mark.asyncio
async def test_client_uses_endpoint_url_and_botocore_config(mocker):
    from agent_platform.core.credentials import ClientOptions

    mock_boto = mocker.patch(
        "agent_platform.integrations.ocr.aws_textract.provider.boto3.client"
    )
    mock_client = MagicMock()
    mock_client.detect_document_text.return_value = FAKE_TEXTRACT_RESPONSE
    mock_boto.return_value = mock_client

    ocr = AWSTextractOCR(
        AWSTextractCredentials(),
        client_options=ClientOptions(
            base_url="https://textract.example.com", timeout=5.0, max_retries=9
        ),
    )

    mocker.patch(
        "agent_platform.integrations.ocr.aws_textract.provider.load_bytes",
        return_value=b"fake-bytes",
    )
    await ocr.extract("a.png")

    _, kwargs = mock_boto.call_args
    assert kwargs["endpoint_url"] == "https://textract.example.com"
    boto_config = kwargs["config"]
    assert boto_config.connect_timeout == 5.0
    assert boto_config.read_timeout == 5.0
    assert boto_config.retries["max_attempts"] == 9


@pytest.mark.asyncio
async def test_client_default_config_sets_max_retries_only(mocker):
    mock_boto = mocker.patch(
        "agent_platform.integrations.ocr.aws_textract.provider.boto3.client"
    )
    mock_client = MagicMock()
    mock_client.detect_document_text.return_value = FAKE_TEXTRACT_RESPONSE
    mock_boto.return_value = mock_client

    ocr = AWSTextractOCR(AWSTextractCredentials())

    mocker.patch(
        "agent_platform.integrations.ocr.aws_textract.provider.load_bytes",
        return_value=b"fake-bytes",
    )
    await ocr.extract("a.png")

    _, kwargs = mock_boto.call_args
    assert "endpoint_url" not in kwargs
    boto_config = kwargs["config"]
    # connect_timeout/read_timeout fall back to botocore's own default (60s)
    # when we don't pass them, since resolve_timeout() returned None.
    assert boto_config.connect_timeout == 60
    assert boto_config.read_timeout == 60
    assert boto_config.retries["max_attempts"] == 3


@pytest.mark.asyncio
async def test_client_is_recreated_for_a_different_region(mocker):
    mock_boto = mocker.patch(
        "agent_platform.integrations.ocr.aws_textract.provider.boto3.client"
    )
    mock_client = MagicMock()
    mock_client.detect_document_text.return_value = FAKE_TEXTRACT_RESPONSE
    mock_boto.return_value = mock_client

    ocr = AWSTextractOCR(AWSTextractCredentials())

    mocker.patch(
        "agent_platform.integrations.ocr.aws_textract.provider.load_bytes",
        return_value=b"fake-bytes",
    )
    await ocr.extract("a.png", config=AWSTextractConfig(region_name="us-east-1"))
    await ocr.extract("b.png", config=AWSTextractConfig(region_name="eu-west-1"))

    assert mock_boto.call_count == 2
