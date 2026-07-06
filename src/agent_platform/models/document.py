from __future__ import annotations

from uuid import UUID, uuid4
from typing import Any
from datetime import datetime

from pydantic import BaseModel, Field

from agent_platform.models.enums import (
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
    width: int | None = None
    height: int | None = None
    color_space: str | None = None
    has_alpha: bool = False
    channels: int | None = None
    bit_depth: int | None = None


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


class VideoDocument(Document, frozen=True):
    media_type: MediaType = MediaType.VIDEO
    content: bytes = b""
    format: VideoFormat = VideoFormat.UNKNOWN
    language: Language | None = None
    duration: float | None = None
    width: int | None = None
    height: int | None = None
    frame_rate: float | None = None
    codec: str | None = None
    bitrate: int | None = None
    has_audio: bool = True
    audio_codec: str | None = None
    audio_channels: int | None = None
    audio_sample_rate: int | None = None
