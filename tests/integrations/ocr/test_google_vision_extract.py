from unittest.mock import MagicMock
from uuid import uuid4

import pytest

pytest.importorskip("google.cloud")

from agent_platform.core.errors import ProviderError
from agent_platform.integrations.ocr.google_vision.config import GoogleVisionConfig
from agent_platform.integrations.ocr.google_vision.google_vision import GoogleVisionOCR
from agent_platform.integrations.ocr.utils import load_bytes


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


async def test_extract_returns_chunks(mocker):
    mock_vision = mocker.patch(
        "agent_platform.integrations.ocr.google_vision.google_vision.vision"
    )
    mock_to_thread = mocker.patch(
        "agent_platform.integrations.ocr.google_vision.google_vision.asyncio.to_thread"
    )
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

    mocker.patch(
        "agent_platform.integrations.ocr.google_vision.google_vision.load_bytes",
        return_value=b"fake-bytes",
    )
    mocker.patch(
        "agent_platform.integrations.ocr.google_vision.google_vision.asyncio.to_thread",
        side_effect=[b"fake-bytes", response],
    )
    result = await ocr.extract("http://example.com/img.jpg", document_id=doc_id)

    assert len(result) == 1
    assert result[0].text == "H"
    assert result[0].document_id == doc_id


async def test_extract_no_pages_returns_empty(mocker):
    mock_vision = mocker.patch(
        "agent_platform.integrations.ocr.google_vision.google_vision.vision"
    )
    mock_client = MagicMock()
    mock_vision.ImageAnnotatorClient.return_value = mock_client
    mock_vision.Image.return_value = MagicMock()

    full_text_annotation = MagicMock()
    full_text_annotation.pages = []
    response = MagicMock()
    response.full_text_annotation = full_text_annotation

    ocr = GoogleVisionOCR()
    mocker.patch(
        "agent_platform.integrations.ocr.google_vision.google_vision.load_bytes",
        return_value=b"fake-bytes",
    )
    mocker.patch(
        "agent_platform.integrations.ocr.google_vision.google_vision.asyncio.to_thread",
        side_effect=[b"fake-bytes", response],
    )
    result = await ocr.extract("http://example.com/img.jpg")
    assert result == []


async def test_extract_filters_by_min_confidence(mocker):
    mock_vision = mocker.patch(
        "agent_platform.integrations.ocr.google_vision.google_vision.vision"
    )
    mock_client = MagicMock()
    mock_vision.ImageAnnotatorClient.return_value = mock_client
    mock_vision.Image.return_value = MagicMock()

    word = _make_word("L", 0.3)
    response = _make_response([[word]])

    ocr = GoogleVisionOCR()
    mocker.patch(
        "agent_platform.integrations.ocr.google_vision.google_vision.load_bytes",
        return_value=b"fake-bytes",
    )
    mocker.patch(
        "agent_platform.integrations.ocr.google_vision.google_vision.asyncio.to_thread",
        side_effect=[b"fake-bytes", response],
    )
    result = await ocr.extract(
        "http://example.com/img.jpg",
        config=GoogleVisionConfig(min_confidence=0.5),
    )
    assert result == []


async def test_extract_raises_on_api_error(mocker):
    mock_vision = mocker.patch(
        "agent_platform.integrations.ocr.google_vision.google_vision.vision"
    )
    mock_client = MagicMock()
    mock_vision.ImageAnnotatorClient.return_value = mock_client

    ocr = GoogleVisionOCR()
    mocker.patch(
        "agent_platform.integrations.ocr.google_vision.google_vision.load_bytes",
        return_value=b"fake-bytes",
    )
    mocker.patch(
        "agent_platform.integrations.ocr.google_vision.google_vision.asyncio.to_thread",
        side_effect=RuntimeError("API error occurred"),
    )
    with pytest.raises(ProviderError, match="API error occurred"):
        await ocr.extract("http://example.com/img.jpg")


async def test_extract_forwards_language_hints_and_uses_document_text_detection(
    mocker,
):
    mock_vision = mocker.patch(
        "agent_platform.integrations.ocr.google_vision.google_vision.vision"
    )
    mock_to_thread = mocker.patch(
        "agent_platform.integrations.ocr.google_vision.google_vision.asyncio.to_thread"
    )
    mock_client = MagicMock()
    mock_vision.ImageAnnotatorClient.return_value = mock_client
    mock_vision.Image.return_value = MagicMock()

    word = _make_word("H", 0.95)
    response = _make_response([[word]])

    calls = []

    async def fake_to_thread(fn, *args, **kwargs):
        calls.append((fn, args, kwargs))
        if fn is load_bytes:
            return b"fake-bytes"
        return response

    mock_to_thread.side_effect = fake_to_thread

    ocr = GoogleVisionOCR()
    mocker.patch(
        "agent_platform.integrations.ocr.google_vision.google_vision.load_bytes",
        return_value=b"fake-bytes",
    )
    await ocr.extract(
        "http://example.com/img.jpg",
        config=GoogleVisionConfig(language_hints=["en"]),
    )

    ocr_call = calls[-1]
    assert ocr_call[0] is mock_client.document_text_detection
    assert ocr_call[2]["image_context"] == {"language_hints": ["en"]}


async def test_extract_uses_text_detection_for_text_detection_feature_type(mocker):
    mock_vision = mocker.patch(
        "agent_platform.integrations.ocr.google_vision.google_vision.vision"
    )
    mock_to_thread = mocker.patch(
        "agent_platform.integrations.ocr.google_vision.google_vision.asyncio.to_thread"
    )
    mock_client = MagicMock()
    mock_vision.ImageAnnotatorClient.return_value = mock_client
    mock_vision.Image.return_value = MagicMock()

    word = _make_word("H", 0.95)
    response = _make_response([[word]])

    calls = []

    async def fake_to_thread(fn, *args, **kwargs):
        calls.append((fn, args, kwargs))
        if fn is load_bytes:
            return b"fake-bytes"
        return response

    mock_to_thread.side_effect = fake_to_thread

    ocr = GoogleVisionOCR()
    mocker.patch(
        "agent_platform.integrations.ocr.google_vision.google_vision.load_bytes",
        return_value=b"fake-bytes",
    )
    await ocr.extract(
        "http://example.com/img.jpg",
        config=GoogleVisionConfig(feature_type="TEXT_DETECTION"),
    )

    ocr_call = calls[-1]
    assert ocr_call[0] is mock_client.text_detection
    assert "image_context" not in ocr_call[2]


async def test_client_is_cached_across_calls(mocker):
    mock_vision = mocker.patch(
        "agent_platform.integrations.ocr.google_vision.google_vision.vision"
    )
    mock_client = MagicMock()
    mock_vision.ImageAnnotatorClient.return_value = mock_client
    mock_vision.Image.return_value = MagicMock()

    response = _make_response([])

    ocr = GoogleVisionOCR()
    mocker.patch(
        "agent_platform.integrations.ocr.google_vision.google_vision.load_bytes",
        return_value=b"fake-bytes",
    )
    mocker.patch(
        "agent_platform.integrations.ocr.google_vision.google_vision.asyncio.to_thread",
        side_effect=[b"fake-bytes", response, b"fake-bytes", response],
    )
    await ocr.extract("http://example.com/img.jpg")
    await ocr.extract("http://example.com/img2.jpg")

    mock_vision.ImageAnnotatorClient.assert_called_once()


async def test_extract_handles_missing_confidence(mocker):
    mock_vision = mocker.patch(
        "agent_platform.integrations.ocr.google_vision.google_vision.vision"
    )
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
    mocker.patch(
        "agent_platform.integrations.ocr.google_vision.google_vision.load_bytes",
        return_value=b"fake-bytes",
    )
    mocker.patch(
        "agent_platform.integrations.ocr.google_vision.google_vision.asyncio.to_thread",
        side_effect=[b"fake-bytes", response],
    )
    result = await ocr.extract("http://example.com/img.jpg")
    assert len(result) == 1
    assert result[0].text == "T"
    assert result[0].confidence.value == 0.0
