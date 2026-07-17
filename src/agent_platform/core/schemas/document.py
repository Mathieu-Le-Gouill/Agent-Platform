from __future__ import annotations

import os
from uuid import UUID, uuid4
from typing import Any
from datetime import datetime

from pydantic import BaseModel, Field

from agent_platform.core.schemas.dimensions import Dimensions
from agent_platform.core.schemas.enums import (
    MediaType,
    DocumentFormat,
    ImageFormat,
    AudioFormat,
    VideoFormat,
    Language,
)


class DocumentMetadata(BaseModel, frozen=True):
    title: str | None = None
    author: str | None = None
    description: str | None = None
    created_at: datetime | None = None
    modified_at: datetime | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel, frozen=True):
    media_type: MediaType
    id: UUID = Field(default_factory=uuid4)
    source: str = ""
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)


class TextDocument(Document, frozen=True):
    media_type: MediaType = MediaType.TEXT
    text: str = ""
    format: DocumentFormat = DocumentFormat.UNKNOWN
    language: Language | None = None
    encoding: str | None = None
    character_count: int | None = None
    word_count: int | None = None
    line_count: int | None = None
    page_count: int | None = None


class ImageDocument(Document, frozen=True):
    media_type: MediaType = MediaType.IMAGE
    content: bytes = b""
    format: ImageFormat = ImageFormat.UNKNOWN
    dimensions: Dimensions | None = None
    color_space: str | None = None
    has_alpha: bool = False
    channels: int | None = None

    @staticmethod
    def load_content(path: str) -> ImageDocument:
        from PIL import Image

        with Image.open(path) as img:
            fmt = _infer_image_format(img)
            width, height = img.size
            has_alpha = img.mode == "RGBA"
            channels = len(img.getbands())
            bit_depth = _image_bit_depth(img.mode)
        with open(path, "rb") as f:
            content = f.read()
        return ImageDocument(
            source=path,
            content=content,
            format=fmt,
            dimensions=Dimensions(width=width, height=height, depth=bit_depth),
            color_space=img.mode,
            has_alpha=has_alpha,
            channels=channels,
            metadata=DocumentMetadata(
                title=os.path.splitext(os.path.basename(path))[0]
            ),
        )

    def save_content(self, path: str) -> None:
        if self.format == ImageFormat.UNKNOWN:
            from PIL import Image as PILImage
            import io

            img = PILImage.open(io.BytesIO(self.content))
            img.save(path)
        else:
            ext = _IMAGE_EXT_MAP.get(self.format, "png")
            full = path if "." in os.path.basename(path) else f"{path}.{ext}"
            with open(full, "wb") as f:
                f.write(self.content)


class AudioDocument(Document, frozen=True):
    media_type: MediaType = MediaType.AUDIO
    content: bytes = b""
    format: AudioFormat = AudioFormat.UNKNOWN
    language: Language | None = None
    duration: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    bitrate: int | None = None
    subtype: str | None = None

    @staticmethod
    def load_content(path: str) -> AudioDocument:
        import soundfile as sf

        info = sf.info(path)
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        fmt = _infer_audio_format(path)
        return AudioDocument(
            source=path,
            content=data.tobytes(),
            format=fmt,
            sample_rate=sr,
            channels=info.channels,
            duration=info.frames / sr,
            subtype=info.subtype,
            metadata=DocumentMetadata(
                title=os.path.splitext(os.path.basename(path))[0]
            ),
        )

    def save_content(self, path: str) -> None:
        import soundfile as sf
        import numpy as np

        arr = np.frombuffer(self.content, dtype=np.float32)
        if self.channels and self.channels > 1:
            arr = arr.reshape(-1, self.channels)
        fmt = _SF_FORMAT_MAP.get(self.format, "WAV")
        sf.write(path, arr, self.sample_rate or 16000, format=fmt)


class VideoDocument(Document, frozen=True):
    media_type: MediaType = MediaType.VIDEO
    content: bytes = b""
    format: VideoFormat = VideoFormat.UNKNOWN
    language: Language | None = None
    duration: float | None = None
    dimensions: Dimensions | None = None
    frame_rate: float | None = None
    codec: str | None = None
    bitrate: int | None = None
    has_audio: bool = True
    audio_codec: str | None = None
    audio_channels: int | None = None
    audio_sample_rate: int | None = None

    @staticmethod
    def load_content(path: str) -> VideoDocument:
        import av

        with av.open(path) as container:
            video_stream = next(
                (s for s in container.streams if s.type == "video"), None
            )
            audio_stream = next(
                (s for s in container.streams if s.type == "audio"), None
            )
            if video_stream is None:
                raise ValueError(f"No video stream found in {path}")
            duration_sec = (
                float(container.duration / av.utils.time_base)
                if container.duration
                else None
            )
            frame_rate = (
                float(video_stream.average_rate) if video_stream.average_rate else None
            )
            codec = video_stream.codec.name if video_stream.codec else None
            video_bitrate = video_stream.bit_rate
            has_audio = audio_stream is not None
            audio_codec = (
                audio_stream.codec.name if audio_stream and audio_stream.codec else None
            )
            audio_channels = audio_stream.channels if audio_stream else None
            audio_sample_rate = audio_stream.rate if audio_stream else None
            fmt = _infer_video_format(path)
        with open(path, "rb") as f:
            content = f.read()
        return VideoDocument(
            source=path,
            content=content,
            format=fmt,
            duration=duration_sec,
            dimensions=Dimensions(width=video_stream.width, height=video_stream.height),
            frame_rate=frame_rate,
            codec=codec,
            bitrate=video_bitrate,
            has_audio=has_audio,
            audio_codec=audio_codec,
            audio_channels=audio_channels,
            audio_sample_rate=audio_sample_rate,
            metadata=DocumentMetadata(
                title=os.path.splitext(os.path.basename(path))[0]
            ),
        )

    def save_content(self, path: str) -> None:
        with open(path, "wb") as f:
            f.write(self.content)


# --- Helpers for load_content / save_content ---

_IMAGE_EXT_MAP: dict[ImageFormat, str] = {
    ImageFormat.JPEG: "jpg",
    ImageFormat.PNG: "png",
    ImageFormat.WEBP: "webp",
    ImageFormat.GIF: "gif",
    ImageFormat.BMP: "bmp",
    ImageFormat.TIFF: "tiff",
}

_SF_FORMAT_MAP: dict[AudioFormat, str] = {
    AudioFormat.WAV: "WAV",
    AudioFormat.FLAC: "FLAC",
    AudioFormat.MP3: "MP3",
    AudioFormat.OGG: "OGG",
    AudioFormat.M4A: "MP4",
}

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


def _image_bit_depth(mode: str) -> int | None:
    return _MODE_BIT_DEPTH.get(mode)


def _infer_image_format(path: str) -> ImageFormat:
    from agent_platform.core.schemas.enums import FileFormat

    ff = FileFormat.from_extension(path)
    if isinstance(ff.extension_format, ImageFormat):
        return ff.extension_format
    return ImageFormat.UNKNOWN


def _infer_audio_format(path: str) -> AudioFormat:
    from agent_platform.core.schemas.enums import FileFormat

    ff = FileFormat.from_extension(path)
    if isinstance(ff.extension_format, AudioFormat):
        return ff.extension_format
    return AudioFormat.UNKNOWN


def _infer_video_format(path: str) -> VideoFormat:
    from agent_platform.core.schemas.enums import FileFormat

    ff = FileFormat.from_extension(path)
    if isinstance(ff.extension_format, VideoFormat):
        return ff.extension_format
    return VideoFormat.UNKNOWN
