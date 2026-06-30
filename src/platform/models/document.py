from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID
from datetime import datetime
from typing import Any

from models.enums.language import Language
from models.enums.fileformat import FileFormat


@dataclass(slots=True, frozen=True)
class DocumentMetadata:
    title:       str | None = None
    author:      str | None = None
    description: str | None = None
    format:      FileFormat | None = None
    created_at:  datetime | None = None
    modified_at: datetime | None = None
    extra:       dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class Document:
    id: UUID
    source: str
    metadata: DocumentMetadata


# --- Text Document ---

@dataclass(slots=True, frozen=True)
class TextInfo:
    mime_type: str | None = None       # text/plain, text/html
    encoding: str | None = None        # UTF-8, UTF-16
    language: Language | None = None


@dataclass(slots=True, frozen=True)
class TextProperties:
    character_count: int | None = None
    word_count: int | None = None
    line_count: int | None = None
    page_count: int | None = None


@dataclass(slots=True, frozen=True)
class TextDocument(Document):
    text: str

    info: TextInfo = field(default_factory=TextInfo)
    properties: TextProperties = field(default_factory=TextProperties)


# --- Image Document ---

@dataclass(slots=True, frozen=True)
class ImageInfo:
    mime_type: str | None = None       # image/png
    ocr_text: str | None = None        # extracted OCR content


@dataclass(slots=True, frozen=True)
class ImageProperties:
    width: int | None = None
    height: int | None = None
    color_space: str | None = None     # RGB, RGBA, L
    has_alpha: bool = False
    channels: int | None = None
    bit_depth: int | None = None


@dataclass(slots=True, frozen=True)
class ImageDocument(Document):
    content: bytes
    info: ImageInfo = field(default_factory=ImageInfo)
    properties: ImageProperties = field(default_factory=ImageProperties)


# --- Audio Document ---

@dataclass(slots=True, frozen=True)
class AudioInfo:
    mime_type: str | None = None       # audio/wav
    language: Language | None = None


@dataclass(slots=True, frozen=True)
class AudioProperties:
    duration_seconds: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    bitrate: int | None = None
    bit_depth: int | None = None


@dataclass(slots=True, frozen=True)
class AudioDocument(Document):
    content: bytes
    info: AudioInfo = field(default_factory=AudioInfo)
    properties: AudioProperties = field(default_factory=AudioProperties)


# --- Video Document ---

@dataclass(slots=True, frozen=True)
class VideoInfo:
    mime_type: str | None = None       # video/mp4
    language: Language | None = None


@dataclass(slots=True, frozen=True)
class VideoProperties:
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    frame_rate: float | None = None
    codec: str | None = None           # H.264, H.265, AV1
    bitrate: int | None = None
    has_audio: bool = True
    audio_codec: str | None = None
    audio_channels: int | None = None
    audio_sample_rate: int | None = None


@dataclass(slots=True, frozen=True)
class VideoDocument(Document):
    content: bytes

    info: VideoInfo = field(default_factory=VideoInfo)
    properties: VideoProperties = field(default_factory=VideoProperties)