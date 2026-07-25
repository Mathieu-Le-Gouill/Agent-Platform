from unittest.mock import MagicMock, mock_open

from agent_platform.core.schemas.dimensions import bit_depth_for_mode as _get_bit_depth
from agent_platform.core.schemas.enums import ImageFormat


class TestGetBitDepth:
    async def test_rgb_8bit(self):
        assert _get_bit_depth("RGB") == 8

    async def test_grayscale_8bit(self):
        assert _get_bit_depth("L") == 8

    async def test_grayscale_16bit(self):
        assert _get_bit_depth("I;16") == 16

    async def test_binary_1bit(self):
        assert _get_bit_depth("1") == 1

    async def test_unknown_returns_none(self):
        assert _get_bit_depth("UNKNOWN") is None


class TestPILImageLoader:
    def _mock_image(self, mocker, mode: str, **overrides):
        mocker.patch("builtins.open", new_callable=mock_open, read_data=b"raw_bytes")
        img = MagicMock()
        img.mode = mode
        img.size = (640, 480)
        img.info = {}
        img.getexif.return_value = {}
        img.getbands.return_value = ("C", "M", "Y", "K")
        for key, value in overrides.items():
            setattr(img, key, value)
        mock_image_cls = mocker.patch(
            "agent_platform.integrations.loader.strategies.pil.pil.Image"
        )
        mock_image_cls.open.return_value.__enter__.return_value = img
        return img

    async def test_load_rgba(self, mocker):
        self._mock_image(mocker, "RGBA")
        from agent_platform.integrations.loader.strategies.pil.pil import (
            PILImageLoader,
        )

        loader = PILImageLoader()
        results = await loader.load("/test/image.png")

        assert len(results) == 1
        doc = results[0]
        assert doc.source == "/test/image.png"
        assert doc.format == ImageFormat.PNG
        assert doc.color_space == "RGBA"
        assert doc.has_alpha is True
        assert doc.channels == 4
        assert doc.content == b"raw_bytes"
        assert doc.dimensions.width == 640
        assert doc.dimensions.height == 480

    async def test_load_rgb(self, mocker):
        self._mock_image(mocker, "RGB")
        from agent_platform.integrations.loader.strategies.pil.pil import (
            PILImageLoader,
        )

        loader = PILImageLoader()
        results = await loader.load("/test/image.jpg")

        assert results[0].color_space == "RGB"
        assert results[0].has_alpha is False
        assert results[0].channels == 3

    async def test_load_grayscale(self, mocker):
        self._mock_image(mocker, "L")
        from agent_platform.integrations.loader.strategies.pil.pil import (
            PILImageLoader,
        )

        loader = PILImageLoader()
        results = await loader.load("/test/image.png")

        assert results[0].color_space == "GRAY"
        assert results[0].has_alpha is False
        assert results[0].channels == 1

    async def test_load_palette_with_transparency(self, mocker):
        self._mock_image(mocker, "P", info={"transparency": 0})
        from agent_platform.integrations.loader.strategies.pil.pil import (
            PILImageLoader,
        )

        loader = PILImageLoader()
        results = await loader.load("/test/image.gif")

        assert results[0].color_space == "PALETTE"
        assert results[0].has_alpha is True
        assert results[0].channels == 4

    async def test_load_palette_without_transparency(self, mocker):
        self._mock_image(mocker, "P")
        from agent_platform.integrations.loader.strategies.pil.pil import (
            PILImageLoader,
        )

        loader = PILImageLoader()
        results = await loader.load("/test/image.gif")

        assert results[0].color_space == "PALETTE"
        assert results[0].has_alpha is False
        assert results[0].channels == 3

    async def test_load_other_mode_falls_back_to_getbands(self, mocker):
        self._mock_image(mocker, "CMYK")
        from agent_platform.integrations.loader.strategies.pil.pil import (
            PILImageLoader,
        )

        loader = PILImageLoader()
        results = await loader.load("/test/image.tiff")

        assert results[0].color_space == "CMYK"
        assert results[0].has_alpha is False
        assert results[0].channels == 4

    async def test_load_unknown_extension_format(self, mocker):
        self._mock_image(mocker, "RGB")
        from agent_platform.integrations.loader.strategies.pil.pil import (
            PILImageLoader,
        )

        loader = PILImageLoader()
        results = await loader.load("/test/image.xyz")

        assert results[0].format == ImageFormat.UNKNOWN

    async def test_load_extracts_exif_title_and_dimensions(self, mocker):
        img = self._mock_image(mocker, "RGB")
        img.getexif.return_value = {256: 1024, 257: 768, 270: "A description"}
        from agent_platform.integrations.loader.strategies.pil.pil import (
            PILImageLoader,
        )

        loader = PILImageLoader()
        results = await loader.load("/test/image.jpg")

        doc = results[0]
        assert doc.metadata.title == "A description"
        assert doc.metadata.extra["width"] == 1024
        assert doc.metadata.extra["height"] == 768
        assert doc.dimensions.width == 1024
        assert doc.dimensions.height == 768

    async def test_load_exif_xp_title_used_when_no_description(self, mocker):
        img = self._mock_image(mocker, "RGB")
        img.getexif.return_value = {40091: "XP Title"}
        from agent_platform.integrations.loader.strategies.pil.pil import (
            PILImageLoader,
        )

        loader = PILImageLoader()
        results = await loader.load("/test/image.jpg")

        assert results[0].metadata.title == "XP Title"

    async def test_load_getexif_raising_is_swallowed(self, mocker):
        img = self._mock_image(mocker, "RGB")
        img.getexif.side_effect = Exception("no exif support")
        from agent_platform.integrations.loader.strategies.pil.pil import (
            PILImageLoader,
        )

        loader = PILImageLoader()
        results = await loader.load("/test/image.jpg")

        assert results[0].metadata.extra == {}
        assert results[0].metadata.title is None
