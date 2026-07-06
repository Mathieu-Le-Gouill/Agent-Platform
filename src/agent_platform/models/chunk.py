from __future__ import annotations

from uuid import UUID, uuid4
from typing import Any

from pydantic import BaseModel, Field

from agent_platform.models.enums import (
    MediaType,
    DocumentFormat,
    AudioFormat,
    VideoFormat,
    DataType,
)


class Chunk(BaseModel, frozen=True):
    media_type: MediaType
    id: UUID = Field(default_factory=uuid4)
    index: int | None = None
    document_id: UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextChunk(Chunk, frozen=True):
    media_type: MediaType = MediaType.TEXT
    text: str = ""
    format: DocumentFormat = DocumentFormat.UNKNOWN
    start_char: int | None = None
    end_char: int | None = None


class AudioChunk(Chunk, frozen=True):
    media_type: MediaType = MediaType.AUDIO
    data: bytes = b""
    sample_rate: int = 16000
    start: int = 0
    end: int = 0
    dtype: DataType = DataType.FLOAT32
    channels: int = 1
    format: AudioFormat = AudioFormat.UNKNOWN


class VideoChunk(Chunk, frozen=True):
    media_type: MediaType = MediaType.VIDEO
    data: bytes = b""
    start: int = 0
    end: int = 0
    width: int | None = None
    height: int | None = None
    frame_rate: float | None = None
    format: VideoFormat = VideoFormat.UNKNOWN
