from __future__ import annotations

import base64

from agent_platform.core.schemas.document import AudioDocument, ImageDocument
from agent_platform.core.schemas.enums import AudioFormat, ImageFormat

__all__ = ["audio_format_name", "image_base64", "image_data_url", "image_mime_type"]

# A document whose format was never detected carries UNKNOWN, a truthy enum
# member; sent as is it becomes `image/unknown` or `unknown`, which vendors
# reject, so it falls back to the most widely accepted format instead.
_DEFAULT_IMAGE_MIME = "image/png"
_DEFAULT_AUDIO_FORMAT = "wav"


def image_mime_type(image: ImageDocument) -> str:
    if image.format is ImageFormat.UNKNOWN:
        return _DEFAULT_IMAGE_MIME
    return f"image/{image.format.value}"


def image_base64(image: ImageDocument) -> str:
    return base64.b64encode(image.content).decode()


def image_data_url(image: ImageDocument) -> str:
    return f"data:{image_mime_type(image)};base64,{image_base64(image)}"


def audio_format_name(audio: AudioDocument) -> str:
    if audio.format is AudioFormat.UNKNOWN:
        return _DEFAULT_AUDIO_FORMAT
    return audio.format.value
