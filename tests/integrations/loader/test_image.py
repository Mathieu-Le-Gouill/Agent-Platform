from unittest.mock import MagicMock, mock_open, patch
from uuid import uuid4

import pytest

from agent_platform.core.interfaces.loader.image.config import ImageLoaderConfig
from agent_platform.core.schemas.enums import ImageFormat


class TestGetBitDepth:
    async def test_rgb_8bit(self):
        from agent_platform.integrations.loader.strategies.pil.pil import (
            _get_bit_depth,
        )

        assert _get_bit_depth("RGB") == 8

    async def test_grayscale_8bit(self):
        from agent_platform.integrations.loader.strategies.pil.pil import (
            _get_bit_depth,
        )

        assert _get_bit_depth("L") == 8

    async def test_grayscale_16bit(self):
        from agent_platform.integrations.loader.strategies.pil.pil import (
            _get_bit_depth,
        )

        assert _get_bit_depth("I;16") == 16

    async def test_binary_1bit(self):
        from agent_platform.integrations.loader.strategies.pil.pil import (
            _get_bit_depth,
        )

        assert _get_bit_depth("1") == 1

    async def test_unknown_returns_none(self):
        from agent_platform.integrations.loader.strategies.pil.pil import (
            _get_bit_depth,
        )

        assert _get_bit_depth("UNKNOWN") is None


class TestExtractExif:
    async def test_empty_exif(self):
        from agent_platform.integrations.loader.strategies.pil.pil import (
            _extract_exif,
        )

        import PIL.Image

        img = MagicMock(spec=PIL.Image.Image)
        img.getexif.return_value = {}
        assert _extract_exif(img) == {}

    async def test_extracts_width_and_height(self):
        from agent_platform.integrations.loader.strategies.pil.pil import (
            _extract_exif,
        )

        import PIL.Image

        img = MagicMock(spec=PIL.Image.Image)
        exif = {0x100: 800, 0x101: 600, 0x010E: "My Photo"}
        img.getexif.return_value = exif
        result = _extract_exif(img)
        assert result["width"] == 800
        assert result["height"] == 600
        assert result["title"] == "My Photo"

    async def test_getexif_exception_returns_empty(self):
        from agent_platform.integrations.loader.strategies.pil.pil import (
            _extract_exif,
        )

        import PIL.Image

        img = MagicMock(spec=PIL.Image.Image)
        img.getexif.side_effect = Exception("corrupt exif")
        assert _extract_exif(img) == {}


class TestPILImageLoader:
    @patch("agent_platform.integrations.loader.strategies.pil.pil.Image")
    @patch("builtins.open", new_callable=mock_open, read_data=b"fake_image_data")
    async def test_load_png(self, mock_file, mock_pil_module, mock_pil_image):
        from agent_platform.integrations.loader.strategies.pil.pil import (
            PILImageLoader,
        )

        mock_img = MagicMock()
        mock_img.mode = "RGBA"
        mock_img.size = (800, 600)
        mock_img.info = {}
        mock_img.getexif.return_value = {}
        mock_pil_module.open.return_value.__enter__.return_value = mock_img

        loader = PILImageLoader()
        results = await loader.load("/test/image.png")

        assert len(results) == 1
        doc = results[0]
        assert doc.source == "/test/image.png"
        assert doc.format == ImageFormat.PNG
        assert doc.dimensions.width == 800
        assert doc.dimensions.height == 600
        assert doc.color_space == "RGBA"
        assert doc.has_alpha is True
        assert doc.channels == 4
        assert doc.dimensions.depth == 8
        assert doc.content == b"fake_image_data"

    @patch("agent_platform.integrations.loader.strategies.pil.pil.Image")
    @patch("builtins.open", new_callable=mock_open, read_data=b"data")
    async def test_load_with_exif_title(
        self, mock_file, mock_pil_module, mock_pil_image
    ):
        from agent_platform.integrations.loader.strategies.pil.pil import (
            PILImageLoader,
        )

        mock_img = MagicMock()
        mock_img.mode = "RGB"
        mock_img.size = (100, 100)
        mock_img.info = {}
        mock_img.getexif.return_value = {0x010E: "Sunset Photo"}
        mock_pil_module.open.return_value.__enter__.return_value = mock_img

        loader = PILImageLoader()
        results = await loader.load("/test/photo.jpg")

        assert results[0].metadata.title == "Sunset Photo"

    @patch("agent_platform.integrations.loader.strategies.pil.pil.Image")
    @patch("builtins.open", new_callable=mock_open, read_data=b"data")
    async def test_load_config_passthrough(
        self, mock_file, mock_pil_module, mock_pil_image
    ):
        from agent_platform.integrations.loader.strategies.pil.pil import (
            PILImageLoader,
        )

        mock_img = MagicMock()
        mock_img.mode = "L"
        mock_img.size = (50, 50)
        mock_img.info = {}
        mock_img.getexif.return_value = {}
        mock_pil_module.open.return_value.__enter__.return_value = mock_img

        loader = PILImageLoader()
        config = ImageLoaderConfig(target_format=ImageFormat.JPEG)
        results = await loader.load("/test/img.png", config=config)

        assert len(results) == 1
        assert results[0].format == ImageFormat.PNG
