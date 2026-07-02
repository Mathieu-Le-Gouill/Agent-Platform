from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from agent_platform.integrations.ocr.providers.google_vision import GoogleVisionOCR
from agent_platform.integrations.ocr.config import GoogleVisionConfig


def _symbol(text):
    return SimpleNamespace(text=text)


def _word(text, confidence):
    return SimpleNamespace(symbols=[_symbol(ch) for ch in text], confidence=confidence)


def _fake_response(words_with_confidence):
    paragraph = SimpleNamespace(
        words=[_word(text, conf) for text, conf in words_with_confidence]
    )
    block = SimpleNamespace(paragraphs=[paragraph])
    page = SimpleNamespace(blocks=[block])
    return SimpleNamespace(
        error=SimpleNamespace(message=""),
        full_text_annotation=SimpleNamespace(pages=[page]),
    )


@pytest.mark.asyncio
async def test_extract_returns_chunks_for_given_document_id():

    document_id = uuid4()
    fake_response = _fake_response([("Hello", 0.9), ("world", 0.8)])

    with patch.object(GoogleVisionOCR, "_build_client") as mock_build_client, \
         patch.object(GoogleVisionOCR, "_load_image", return_value="fake-image"):

        mock_client = MagicMock()
        mock_client.annotate_image.return_value = fake_response
        mock_build_client.return_value = mock_client

        ocr = GoogleVisionOCR(GoogleVisionConfig())
        chunks = await ocr.extract("some/path.png", document_id=document_id)

    assert [c.text for c in chunks] == ["Hello world"]
    assert all(c.document_id == document_id for c in chunks)
    mock_client.annotate_image.assert_called_once()


@pytest.mark.asyncio
async def test_extract_raises_on_api_error():

    fake_response = SimpleNamespace(
        error=SimpleNamespace(message="quota exceeded"),
        full_text_annotation=SimpleNamespace(pages=[]),
    )

    with patch.object(GoogleVisionOCR, "_build_client") as mock_build_client, \
         patch.object(GoogleVisionOCR, "_load_image", return_value="fake-image"):

        mock_client = MagicMock()
        mock_client.annotate_image.return_value = fake_response
        mock_build_client.return_value = mock_client

        ocr = GoogleVisionOCR(GoogleVisionConfig())

        with pytest.raises(RuntimeError, match="quota exceeded"):
            await ocr.extract("some/path.png")


@pytest.mark.asyncio
async def test_extract_uses_configured_feature_type():

    fake_response = _fake_response([("Hi", 0.99)])
    captured_request = {}

    def fake_annotate_image(request):
        captured_request["request"] = request
        return fake_response

    with patch.object(GoogleVisionOCR, "_build_client") as mock_build_client, \
         patch.object(GoogleVisionOCR, "_load_image", return_value="fake-image"):

        mock_client = MagicMock()
        mock_client.annotate_image.side_effect = fake_annotate_image
        mock_build_client.return_value = mock_client

        ocr = GoogleVisionOCR(GoogleVisionConfig(feature_type="TEXT_DETECTION"))
        await ocr.extract("some/path.png")

    feature = captured_request["request"].features[0]
    assert feature.type_ == GoogleVisionOCR._feature_type(
        GoogleVisionConfig(feature_type="TEXT_DETECTION")
    )