from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from agent_platform.integrations.ocr.config import GoogleVisionConfig
from agent_platform.models.chunk import TextChunk


@patch("agent_platform.integrations.ocr.providers.google_vision.vision")
async def test_extract_returns_chunks(mock_vision):
    mock_symbol = MagicMock()
    mock_symbol.text = "H"
    mock_symbol.confidence = 0.95
    mock_word = MagicMock()
    mock_word.symbols = [mock_symbol]
    mock_paragraph = MagicMock()
    mock_paragraph.words = [mock_word]
    mock_block = MagicMock()
    mock_block.paragraphs = [mock_paragraph]
    mock_page = MagicMock()
    mock_page.blocks = [mock_block]
    mock_page.page_number = 1
    mock_annotation = MagicMock()
    mock_annotation.pages = [mock_page]
    mock_response = MagicMock()
    mock_response.full_text_annotation = mock_annotation
    mock_response.error.message = ""

    mock_client = MagicMock()
    mock_client.annotate_image.return_value = mock_response
    mock_vision.ImageAnnotatorClient.return_value = mock_client

    from agent_platform.integrations.ocr.providers.google_vision import GoogleVisionOCR

    ocr = GoogleVisionOCR()

    doc_id = uuid4()
    result = await ocr.extract("http://example.com/img.jpg", document_id=doc_id)

    assert len(result) == 1
    assert result[0].text == "H"
    assert result[0].document_id == doc_id


@patch("agent_platform.integrations.ocr.providers.google_vision.vision")
async def test_extract_no_pages_returns_empty(mock_vision):
    mock_response = MagicMock()
    mock_response.full_text_annotation.pages = []
    mock_response.error.message = ""

    mock_client = MagicMock()
    mock_client.annotate_image.return_value = mock_response
    mock_vision.ImageAnnotatorClient.return_value = mock_client

    from agent_platform.integrations.ocr.providers.google_vision import GoogleVisionOCR

    ocr = GoogleVisionOCR()

    result = await ocr.extract("http://example.com/img.jpg")
    assert result == []


@patch("agent_platform.integrations.ocr.providers.google_vision.vision")
async def test_extract_raises_on_api_error(mock_vision):
    mock_response = MagicMock()
    mock_response.full_text_annotation.pages = []
    mock_response.error.message = "API error occurred"

    mock_client = MagicMock()
    mock_client.annotate_image.return_value = mock_response
    mock_vision.ImageAnnotatorClient.return_value = mock_client

    from agent_platform.integrations.ocr.providers.google_vision import GoogleVisionOCR

    ocr = GoogleVisionOCR()

    with pytest.raises(RuntimeError, match="API error occurred"):
        await ocr.extract("http://example.com/img.jpg")


@patch("agent_platform.integrations.ocr.providers.google_vision.vision")
async def test_extract_filters_by_min_confidence(mock_vision):
    mock_low_symbol = MagicMock()
    mock_low_symbol.text = "L"
    mock_low_symbol.confidence = 0.3
    mock_low_word = MagicMock()
    mock_low_word.symbols = [mock_low_symbol]
    mock_low_paragraph = MagicMock()
    mock_low_paragraph.words = [mock_low_word]
    mock_low_block = MagicMock()
    mock_low_block.paragraphs = [mock_low_paragraph]
    mock_page = MagicMock()
    mock_page.blocks = [mock_low_block]
    mock_page.page_number = 1
    mock_annotation = MagicMock()
    mock_annotation.pages = [mock_page]
    mock_response = MagicMock()
    mock_response.full_text_annotation = mock_annotation
    mock_response.error.message = ""

    mock_client = MagicMock()
    mock_client.annotate_image.return_value = mock_response
    mock_vision.ImageAnnotatorClient.return_value = mock_client

    from agent_platform.integrations.ocr.providers.google_vision import GoogleVisionOCR

    ocr = GoogleVisionOCR()

    result = await ocr.extract(
        "http://example.com/img.jpg",
        config=GoogleVisionConfig(min_confidence=0.5),
    )
    assert result == []


@patch("agent_platform.integrations.ocr.providers.google_vision.vision")
async def test_extract_handles_missing_confidence(mock_vision):
    mock_symbol = MagicMock()
    mock_symbol.text = "T"
    mock_symbol.confidence = None
    mock_word = MagicMock()
    mock_word.symbols = [mock_symbol]
    mock_paragraph = MagicMock()
    mock_paragraph.words = [mock_word]
    mock_block = MagicMock()
    mock_block.paragraphs = [mock_paragraph]
    mock_page = MagicMock()
    mock_page.blocks = [mock_block]
    mock_page.page_number = 1
    mock_annotation = MagicMock()
    mock_annotation.pages = [mock_page]
    mock_response = MagicMock()
    mock_response.full_text_annotation = mock_annotation
    mock_response.error.message = ""

    mock_client = MagicMock()
    mock_client.annotate_image.return_value = mock_response
    mock_vision.ImageAnnotatorClient.return_value = mock_client

    from agent_platform.integrations.ocr.providers.google_vision import GoogleVisionOCR

    ocr = GoogleVisionOCR()

    result = await ocr.extract("http://example.com/img.jpg")
    assert len(result) == 1
    assert result[0].text == "T"
    assert result[0].metadata["confidence"] == 0.0
