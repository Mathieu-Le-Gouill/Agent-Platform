from unittest.mock import MagicMock, mock_open, patch
from uuid import uuid4

from agent_platform.integrations.ocr.config import (
    AWSTextractConfig,
    GoogleVisionConfig,
    TesseractConfig,
)
from agent_platform.integrations.ocr.providers.aws_textract import AWSTextractOCR
from agent_platform.integrations.ocr.providers.google_vision import (
    GoogleVisionOCR,
    _feature_type,
)
from agent_platform.integrations.ocr.providers.tesseract import TesseractOCR


class TestGoogleVisionLoadImage:
    @patch("agent_platform.integrations.ocr.providers.google_vision.vision.Image")
    @patch("agent_platform.integrations.ocr.providers.google_vision.vision.ImageSource")
    def test_url_source(self, MockImageSource, MockImage):
        source = "https://example.com/image.jpg"
        GoogleVisionOCR._load_image(source)
        MockImageSource.assert_called_once_with(image_uri=source)
        MockImage.assert_called_once_with(source=MockImageSource.return_value)

    @patch("agent_platform.integrations.ocr.providers.google_vision.vision.Image")
    def test_file_path(self, MockImage):
        fake_bytes = b"fake-image-content"
        with patch("builtins.open", mock_open(read_data=fake_bytes)):
            GoogleVisionOCR._load_image("/path/to/image.png")
        MockImage.assert_called_once_with(content=fake_bytes)


class TestGoogleVisionBuildClient:
    @patch(
        "agent_platform.integrations.ocr.providers.google_vision."
        "vision.ImageAnnotatorClient.from_service_account_file"
    )
    def test_with_credentials_path(self, mock_from_file):
        mock_from_file.return_value = "client-with-creds"
        config = GoogleVisionConfig(credentials_path="/path/to/creds.json")
        ocr = GoogleVisionOCR(config)
        mock_from_file.assert_called_once_with("/path/to/creds.json")
        assert ocr._client == "client-with-creds"

    @patch(
        "agent_platform.integrations.ocr.providers.google_vision."
        "vision.ImageAnnotatorClient"
    )
    def test_without_credentials_path(self, MockClient):
        MockClient.return_value = "client-no-creds"
        config = GoogleVisionConfig()
        ocr = GoogleVisionOCR(config)
        MockClient.assert_called_once_with()
        assert ocr._client == "client-no-creds"


class TestFeatureTypeSpecific:
    def test_known_name_returns_correct_enum(self):
        from google.cloud.vision import Feature

        result = _feature_type("DOCUMENT_TEXT_DETECTION")
        assert result == Feature.Type.DOCUMENT_TEXT_DETECTION

    def test_unknown_name_falls_back_to_document_text_detection(self):
        from google.cloud.vision import Feature

        result = _feature_type("NONEXISTENT")
        assert result == Feature.Type.DOCUMENT_TEXT_DETECTION


class TestAWSTextractLoadBytes:
    def test_url_source(self):
        source = "https://example.com/document.png"
        fake_bytes = b"fake-image-bytes"
        with patch(
            "agent_platform.integrations.ocr.providers.aws_textract.urlopen"
        ) as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = fake_bytes
            mock_urlopen.return_value.__enter__.return_value = mock_response
            result = AWSTextractOCR._load_bytes(source)
        assert result == fake_bytes
        mock_urlopen.assert_called_once_with(source)

    def test_file_path(self):
        fake_bytes = b"fake-file-bytes"
        with patch("builtins.open", mock_open(read_data=fake_bytes)):
            result = AWSTextractOCR._load_bytes("/path/to/doc.png")
        assert result == fake_bytes


class TestTesseractLoadImage:
    @patch("agent_platform.integrations.ocr.providers.tesseract.Image")
    @patch("agent_platform.integrations.ocr.providers.tesseract.urlopen")
    def test_url_source(self, mock_urlopen, MockImage):
        source = "https://example.com/image.png"
        mock_response = MagicMock()
        mock_urlopen.return_value.__enter__.return_value = mock_response
        MockImage.open.return_value.convert.return_value = "fake-rgb-image"
        result = TesseractOCR._load_image(source)
        MockImage.open.assert_called_once_with(mock_response)
        MockImage.open.return_value.convert.assert_called_once_with("RGB")
        assert result == "fake-rgb-image"

    @patch("agent_platform.integrations.ocr.providers.tesseract.Image")
    @patch("agent_platform.integrations.ocr.providers.tesseract.Path")
    def test_file_path(self, MockPath, MockImage):
        source = "/path/to/image.png"
        MockImage.open.return_value.convert.return_value = "fake-rgb-image"
        result = TesseractOCR._load_image(source)
        MockPath.assert_called_once_with(source)
        MockImage.open.assert_called_once_with(MockPath.return_value)
        MockImage.open.return_value.convert.assert_called_once_with("RGB")
        assert result == "fake-rgb-image"
