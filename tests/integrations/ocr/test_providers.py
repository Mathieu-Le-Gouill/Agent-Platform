from unittest.mock import MagicMock, mock_open

import pytest

pytest.importorskip("boto3")
pytest.importorskip("google.cloud")
pytest.importorskip("pytesseract")

from agent_platform.integrations.ocr.sources import load_bytes
from agent_platform.integrations.ocr.tesseract.provider import TesseractOCR


class TestOCRLoadBytes:
    def test_url_source(self, mocker):
        source = "https://example.com/image.jpg"
        fake_bytes = b"fake-image-bytes"
        mock_urlopen = mocker.patch("agent_platform.integrations.ocr.sources.urlopen")
        mock_response = MagicMock()
        mock_response.read.return_value = fake_bytes
        mock_urlopen.return_value.__enter__.return_value = mock_response
        result = load_bytes(source)
        assert result == fake_bytes
        mock_urlopen.assert_called_once_with(source)

    def test_file_path(self, mocker):
        fake_bytes = b"fake-image-content"
        mocker.patch("builtins.open", mock_open(read_data=fake_bytes))
        result = load_bytes("/path/to/image.png")
        assert result == fake_bytes


class TestTesseractLoadImage:
    def test_url_source(self, mocker):
        mock_urlopen = mocker.patch(
            "agent_platform.integrations.ocr.tesseract.provider.urlopen"
        )
        MockImage = mocker.patch(
            "agent_platform.integrations.ocr.tesseract.provider.Image"
        )
        source = "https://example.com/image.png"
        mock_response = MagicMock()
        mock_urlopen.return_value.__enter__.return_value = mock_response
        MockImage.open.return_value.convert.return_value = "fake-rgb-image"
        result = TesseractOCR._load_image(source)
        MockImage.open.assert_called_once_with(mock_response)
        MockImage.open.return_value.convert.assert_called_once_with("RGB")
        assert result == "fake-rgb-image"

    def test_file_path(self, mocker):
        MockPath = mocker.patch(
            "agent_platform.integrations.ocr.tesseract.provider.Path"
        )
        MockImage = mocker.patch(
            "agent_platform.integrations.ocr.tesseract.provider.Image"
        )
        source = "/path/to/image.png"
        MockImage.open.return_value.convert.return_value = "fake-rgb-image"
        result = TesseractOCR._load_image(source)
        MockPath.assert_called_once_with(source)
        MockImage.open.assert_called_once_with(MockPath.return_value)
        MockImage.open.return_value.convert.assert_called_once_with("RGB")
        assert result == "fake-rgb-image"
