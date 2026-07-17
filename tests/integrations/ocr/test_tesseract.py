from unittest.mock import patch
from uuid import uuid4

import pytest

pytest.importorskip("pytesseract")

from agent_platform.integrations.ocr.tesseract.tesseract import TesseractOCR
from agent_platform.integrations.ocr.tesseract.config import TesseractConfig


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

    ocr = TesseractOCR()
    document_id = uuid4()

    with (
        patch.object(TesseractOCR, "_load_image", return_value="fake-image"),
        patch(
            "agent_platform.integrations.ocr.tesseract.tesseract.pytesseract.image_to_data",
            return_value=FAKE_TESSERACT_DATA,
        ),
    ):
        chunks = await ocr.extract("some/path.png", document_id=document_id)

    assert [c.text for c in chunks] == ["Total:", "42.00"]
    assert all(c.document_id == document_id for c in chunks)


@pytest.mark.asyncio
async def test_extract_mints_document_id_when_not_provided():

    ocr = TesseractOCR()

    with (
        patch.object(TesseractOCR, "_load_image", return_value="fake-image"),
        patch(
            "agent_platform.integrations.ocr.tesseract.tesseract.pytesseract.image_to_data",
            return_value=FAKE_TESSERACT_DATA,
        ),
    ):
        chunks = await ocr.extract("some/path.png")

    assert chunks[0].document_id is not None


@pytest.mark.asyncio
async def test_extract_applies_min_confidence_from_config():

    ocr = TesseractOCR()

    with (
        patch.object(TesseractOCR, "_load_image", return_value="fake-image"),
        patch(
            "agent_platform.integrations.ocr.tesseract.tesseract.pytesseract.image_to_data",
            return_value=FAKE_TESSERACT_DATA,
        ),
    ):
        chunks = await ocr.extract(
            "some/path.png", config=TesseractConfig(min_confidence=90.0)
        )

    # only "Total:" (conf=91) clears the 90 threshold, "42.00" (conf=87) is dropped
    assert [c.text for c in chunks] == ["Total:"]


@patch("agent_platform.integrations.ocr.tesseract.tesseract.pytesseract")
async def test_custom_tesseract_cmd(mock_pytesseract):
    mock_pytesseract.image_to_data.return_value = FAKE_TESSERACT_DATA
    mock_pytesseract.Output.DICT = "dict"
    config = TesseractConfig(tesseract_cmd="/usr/local/bin/tesseract")
    ocr = TesseractOCR()
    with patch.object(TesseractOCR, "_load_image", return_value="fake-image"):
        await ocr.extract("some/path.png", config=config)
    assert mock_pytesseract.pytesseract.tesseract_cmd == "/usr/local/bin/tesseract"
