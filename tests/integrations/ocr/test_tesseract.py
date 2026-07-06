from unittest.mock import patch
from uuid import uuid4

import pytest

from agent_platform.integrations.ocr.providers.tesseract import TesseractOCR
from agent_platform.integrations.ocr.config import TesseractConfig


FAKE_TESSERACT_DATA = {
    "text": ["Total:", "42.00"],
    "conf": ["91", "87"],
    "left": [0, 60],
    "top": [0, 0],
    "width": [50, 40],
    "height": [12, 12],
    "page_num": [1, 1],
}


@pytest.mark.asyncio
async def test_extract_returns_chunks_for_given_document_id():

    ocr = TesseractOCR(TesseractConfig())
    document_id = uuid4()

    with (
        patch.object(TesseractOCR, "_load_image", return_value="fake-image"),
        patch(
            "agent_platform.integrations.ocr.providers.tesseract.pytesseract.image_to_data",
            return_value=FAKE_TESSERACT_DATA,
        ),
    ):
        chunks = await ocr.extract("some/path.png", document_id=document_id)

    assert [c.text for c in chunks] == ["Total:", "42.00"]
    assert all(c.document_id == document_id for c in chunks)


@pytest.mark.asyncio
async def test_extract_mints_document_id_when_not_provided():

    ocr = TesseractOCR(TesseractConfig())

    with (
        patch.object(TesseractOCR, "_load_image", return_value="fake-image"),
        patch(
            "agent_platform.integrations.ocr.providers.tesseract.pytesseract.image_to_data",
            return_value=FAKE_TESSERACT_DATA,
        ),
    ):
        chunks = await ocr.extract("some/path.png")

    assert chunks[0].document_id is not None


@pytest.mark.asyncio
async def test_extract_applies_min_confidence_from_config():

    ocr = TesseractOCR(TesseractConfig(min_confidence=90.0))

    with (
        patch.object(TesseractOCR, "_load_image", return_value="fake-image"),
        patch(
            "agent_platform.integrations.ocr.providers.tesseract.pytesseract.image_to_data",
            return_value=FAKE_TESSERACT_DATA,
        ),
    ):
        chunks = await ocr.extract("some/path.png")

    # only "Total:" (conf=91) clears the 90 threshold, "42.00" (conf=87) is dropped
    assert [c.text for c in chunks] == ["Total:"]


@patch("agent_platform.integrations.ocr.providers.tesseract.pytesseract")
def test_custom_tesseract_cmd(mock_pytesseract):
    config = TesseractConfig(tesseract_cmd="/usr/local/bin/tesseract")
    TesseractOCR(config)
    assert mock_pytesseract.pytesseract.tesseract_cmd == "/usr/local/bin/tesseract"
