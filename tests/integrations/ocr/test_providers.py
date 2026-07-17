from unittest.mock import MagicMock, mock_open, patch
from uuid import uuid4

import pytest

pytest.importorskip("boto3")
pytest.importorskip("google.cloud")
pytest.importorskip("pytesseract")

from agent_platform.integrations.ocr.aws_textract.config import AWSTextractConfig
from agent_platform.integrations.ocr.google_vision.config import GoogleVisionConfig
from agent_platform.integrations.ocr.tesseract.config import TesseractConfig
from agent_platform.integrations.ocr.aws_textract.aws_textract import AWSTextractOCR
from agent_platform.integrations.ocr.google_vision.google_vision import GoogleVisionOCR
from agent_platform.integrations.ocr.tesseract.tesseract import TesseractOCR
from agent_platform.integrations.ocr.utils import load_bytes


class TestOCRLoadBytes:
    def test_url_source(self):
        source = "https://example.com/image.jpg"
        fake_bytes = b"fake-image-bytes"
        with patch("agent_platform.integrations.ocr.utils.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = fake_bytes
            mock_urlopen.return_value.__enter__.return_value = mock_response
            result = load_bytes(source)
        assert result == fake_bytes
        mock_urlopen.assert_called_once_with(source)

    def test_file_path(self):
        fake_bytes = b"fake-image-content"
        with patch("builtins.open", mock_open(read_data=fake_bytes)):
            result = load_bytes("/path/to/image.png")
        assert result == fake_bytes


class TestTesseractLoadImage:
    @patch("agent_platform.integrations.ocr.tesseract.tesseract.Image")
    @patch("agent_platform.integrations.ocr.tesseract.tesseract.urlopen")
    def test_url_source(self, mock_urlopen, MockImage):
        source = "https://example.com/image.png"
        mock_response = MagicMock()
        mock_urlopen.return_value.__enter__.return_value = mock_response
        MockImage.open.return_value.convert.return_value = "fake-rgb-image"
        result = TesseractOCR._load_image(source)
        MockImage.open.assert_called_once_with(mock_response)
        MockImage.open.return_value.convert.assert_called_once_with("RGB")
        assert result == "fake-rgb-image"

    @patch("agent_platform.integrations.ocr.tesseract.tesseract.Image")
    @patch("agent_platform.integrations.ocr.tesseract.tesseract.Path")
    def test_file_path(self, MockPath, MockImage):
        source = "/path/to/image.png"
        MockImage.open.return_value.convert.return_value = "fake-rgb-image"
        result = TesseractOCR._load_image(source)
        MockPath.assert_called_once_with(source)
        MockImage.open.assert_called_once_with(MockPath.return_value)
        MockImage.open.return_value.convert.assert_called_once_with("RGB")
        assert result == "fake-rgb-image"
