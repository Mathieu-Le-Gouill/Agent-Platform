from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch
from uuid import uuid4

import pytest

pytest.importorskip("google.cloud")

from agent_platform.integrations.ocr.google_vision.config import GoogleVisionConfig
from agent_platform.integrations.ocr.google_vision.google_vision import GoogleVisionOCR
from agent_platform.integrations.ocr.utils import load_bytes
from agent_platform.integrations.credentials import (
    GoogleVisionCredentials,
)
from agent_platform.core.errors import ProviderError
from agent_platform.core.schemas.chunk import TextChunk


def _make_response(words_per_paragraph):
    paragraphs = []
    for words in words_per_paragraph:
        paragraph = MagicMock()
        paragraph.words = words
        paragraphs.append(paragraph)
    block = MagicMock()
    block.paragraphs = paragraphs
    page = MagicMock()
    page.blocks = [block]
    full_text_annotation = MagicMock()
    full_text_annotation.pages = [page]
    response = MagicMock()
    response.full_text_annotation = full_text_annotation
    return response


def _make_word(text, confidence):
    symbols = []
    for ch in text:
        sym = MagicMock()
        sym.text = ch
        sym.confidence = confidence
        symbols.append(sym)
    word = MagicMock()
    word.symbols = symbols
    return word


@patch("agent_platform.integrations.ocr.google_vision.google_vision.asyncio.to_thread")
@patch("agent_platform.integrations.ocr.google_vision.google_vision.vision")
async def test_extract_returns_chunks(mock_vision, mock_to_thread):
    mock_client = MagicMock()
    mock_vision.ImageAnnotatorClient.return_value = mock_client
    mock_vision.Image.return_value = MagicMock()

    word = _make_word("H", 0.95)
    response = _make_response([[word]])

    async def fake_to_thread(fn, *args, **kwargs):
        if fn == load_bytes:
            return b"fake-bytes"
        return fn(*args, **kwargs) if args else response

    mock_to_thread.side_effect = fake_to_thread

    # Patch _load_bytes directly
    ocr = GoogleVisionOCR()
    doc_id = uuid4()

    with patch("agent_platform.integrations.ocr.google_vision.google_vision.load_bytes", return_value=b"fake-bytes"):
        with patch(
            "agent_platform.integrations.ocr.google_vision.google_vision.asyncio.to_thread",
            side_effect=[b"fake-bytes", response],
        ):
            result = await ocr.extract("http://example.com/img.jpg", document_id=doc_id)

    assert len(result) == 1
    assert result[0].text == "H"
    assert result[0].document_id == doc_id


@patch("agent_platform.integrations.ocr.google_vision.google_vision.vision")
async def test_extract_no_pages_returns_empty(mock_vision):
    mock_client = MagicMock()
    mock_vision.ImageAnnotatorClient.return_value = mock_client
    mock_vision.Image.return_value = MagicMock()

    full_text_annotation = MagicMock()
    full_text_annotation.pages = []
    response = MagicMock()
    response.full_text_annotation = full_text_annotation

    ocr = GoogleVisionOCR()
    with patch("agent_platform.integrations.ocr.google_vision.google_vision.load_bytes", return_value=b"fake-bytes"):
        with patch(
            "agent_platform.integrations.ocr.google_vision.google_vision.asyncio.to_thread",
            side_effect=[b"fake-bytes", response],
        ):
            result = await ocr.extract("http://example.com/img.jpg")
    assert result == []


@patch("agent_platform.integrations.ocr.google_vision.google_vision.vision")
async def test_extract_filters_by_min_confidence(mock_vision):
    mock_client = MagicMock()
    mock_vision.ImageAnnotatorClient.return_value = mock_client
    mock_vision.Image.return_value = MagicMock()

    word = _make_word("L", 0.3)
    response = _make_response([[word]])

    ocr = GoogleVisionOCR()
    with patch("agent_platform.integrations.ocr.google_vision.google_vision.load_bytes", return_value=b"fake-bytes"):
        with patch(
            "agent_platform.integrations.ocr.google_vision.google_vision.asyncio.to_thread",
            side_effect=[b"fake-bytes", response],
        ):
            result = await ocr.extract(
                "http://example.com/img.jpg",
                config=GoogleVisionConfig(min_confidence=0.5),
            )
    assert result == []


@patch("agent_platform.integrations.ocr.google_vision.google_vision.vision")
async def test_extract_raises_on_api_error(mock_vision):
    mock_client = MagicMock()
    mock_vision.ImageAnnotatorClient.return_value = mock_client

    ocr = GoogleVisionOCR()
    with patch("agent_platform.integrations.ocr.google_vision.google_vision.load_bytes", return_value=b"fake-bytes"):
        with patch(
            "agent_platform.integrations.ocr.google_vision.google_vision.asyncio.to_thread",
            side_effect=RuntimeError("API error occurred"),
        ):
            with pytest.raises(ProviderError, match="API error occurred"):
                await ocr.extract("http://example.com/img.jpg")


@patch("agent_platform.integrations.ocr.google_vision.google_vision.vision")
async def test_extract_handles_missing_confidence(mock_vision):
    mock_client = MagicMock()
    mock_vision.ImageAnnotatorClient.return_value = mock_client
    mock_vision.Image.return_value = MagicMock()

    sym = MagicMock()
    sym.text = "T"
    sym.confidence = None
    word = MagicMock()
    word.symbols = [sym]
    response = _make_response([[word]])

    ocr = GoogleVisionOCR()
    with patch("agent_platform.integrations.ocr.google_vision.google_vision.load_bytes", return_value=b"fake-bytes"):
        with patch(
            "agent_platform.integrations.ocr.google_vision.google_vision.asyncio.to_thread",
            side_effect=[b"fake-bytes", response],
        ):
            result = await ocr.extract("http://example.com/img.jpg")
    assert len(result) == 1
    assert result[0].text == "T"
    assert result[0].confidence.value == 0.0
