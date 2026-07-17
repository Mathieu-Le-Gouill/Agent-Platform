from typing import Sequence
from uuid import uuid4

from PIL import Image
from PIL.ExifTags import TAGS as EXIF_TAGS

from agent_platform.core.interfaces.loader.image.base import BaseImageLoader
from agent_platform.core.interfaces.loader.image.config import ImageLoaderConfig
from agent_platform.core.schemas.document import ImageDocument, DocumentMetadata
from agent_platform.core.schemas.enums import ImageFormat, FileFormat


class PILImageLoader(BaseImageLoader):
    async def load(
        self, source: str, config: ImageLoaderConfig | None = None
    ) -> Sequence[ImageDocument]:
        ff = FileFormat.from_path(source)
        if isinstance(ff.extension_format, ImageFormat):
            fmt = ff.extension_format
        else:
            fmt = ImageFormat.UNKNOWN

        with Image.open(source) as img:
            if img.mode == "RGBA":
                has_alpha = True
                color_space = "RGBA"
                channels = 4
            elif img.mode == "RGB":
                has_alpha = False
                color_space = "RGB"
                channels = 3
            elif img.mode == "L":
                has_alpha = False
                color_space = "GRAY"
                channels = 1
            elif img.mode == "P":
                has_alpha = "transparency" in img.info
                color_space = "PALETTE"
                channels = 4 if has_alpha else 3
            else:
                has_alpha = False
                color_space = img.mode
                channels = len(img.getbands())

            bit_depth = _get_bit_depth(img.mode)
            img_width, img_height = img.size

            exif_data = _extract_exif(img)

        with open(source, "rb") as f:
            raw_bytes = f.read()

        return [
            ImageDocument(
                id=uuid4(),
                source=source,
                metadata=DocumentMetadata(
                    title=exif_data.get("title") or exif_data.get("ImageDescription"),
                    extra=exif_data,
                ),
                content=raw_bytes,
                format=fmt,
                width=exif_data.get("width") or img_width,
                height=exif_data.get("height") or img_height,
                color_space=color_space,
                has_alpha=has_alpha,
                channels=channels,
                bit_depth=bit_depth,
            )
        ]


_MODE_BIT_DEPTH: dict[str, int] = {
    "1": 1,
    "L": 8,
    "LA": 8,
    "P": 8,
    "I": 32,
    "F": 32,
    "RGB": 8,
    "RGBA": 8,
    "CMYK": 8,
    "YCbCr": 8,
    "LAB": 8,
    "HSV": 8,
    "I;16": 16,
    "I;16L": 16,
    "I;16B": 16,
}


def _get_bit_depth(mode: str) -> int | None:
    return _MODE_BIT_DEPTH.get(mode)


def _extract_exif(img: Image.Image) -> dict:
    result: dict = {}
    try:
        exif = img.getexif()
    except Exception:
        return result
    for tag_id, value in exif.items():
        tag_name = EXIF_TAGS.get(tag_id, str(tag_id))
        if tag_name in ("ImageWidth", "PixelXDimension"):
            result["width"] = int(value)
        elif tag_name in ("ImageLength", "PixelYDimension"):
            result["height"] = int(value)
        elif tag_name in ("ImageDescription", "XPTitle"):
            result["title"] = str(value)
    return result
