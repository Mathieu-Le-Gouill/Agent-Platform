from unittest.mock import MagicMock, mock_open
from urllib.error import HTTPError

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
        mock_open_url = mocker.patch("agent_platform.integrations.ocr.sources.open_url")
        mock_response = MagicMock()
        mock_response.read.return_value = fake_bytes
        mock_open_url.return_value.__enter__.return_value = mock_response
        result = load_bytes(source)
        assert result == fake_bytes
        mock_open_url.assert_called_once_with(source)

    def test_file_path(self, mocker):
        fake_bytes = b"fake-image-content"
        mocker.patch("builtins.open", mock_open(read_data=fake_bytes))
        result = load_bytes("/path/to/image.png")
        assert result == fake_bytes


class TestTesseractLoadImage:
    def test_url_source(self, mocker):
        mock_open_url = mocker.patch(
            "agent_platform.integrations.ocr.tesseract.provider.open_url"
        )
        MockImage = mocker.patch(
            "agent_platform.integrations.ocr.tesseract.provider.Image"
        )
        source = "https://example.com/image.png"
        mock_response = MagicMock()
        mock_open_url.return_value.__enter__.return_value = mock_response
        MockImage.open.return_value.convert.return_value = "fake-rgb-image"
        result = TesseractOCR._load_image(source)
        mock_open_url.assert_called_once_with(source)
        MockImage.open.assert_called_once_with(mock_response)
        MockImage.open.return_value.convert.assert_called_once_with("RGB")
        assert result == "fake-rgb-image"

    def test_url_redirect_is_refused(self, http_server):
        http_server.redirect("/image.png", 302, "/target")
        http_server.ok("/target", b"target-bytes")

        with pytest.raises(HTTPError) as excinfo:
            TesseractOCR._load_image(http_server.url("/image.png"))

        assert excinfo.value.code == 302
        assert "/target" not in http_server.requested

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
