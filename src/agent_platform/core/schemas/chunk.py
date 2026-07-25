from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import Field

from agent_platform.core.base import Entity
from agent_platform.core.schemas.bounding_box import BoundingBox
from agent_platform.core.schemas.dimensions import Dimensions
from agent_platform.core.schemas.enums import (
    AudioFormat,
    DataType,
    DocumentFormat,
    MediaType,
    VideoFormat,
)
from agent_platform.core.schemas.score import Score


class Chunk(Entity, frozen=True):
    media_type: MediaType
    index: int | None = None
    document_id: UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextChunk(Chunk, frozen=True):
    media_type: MediaType = MediaType.TEXT
    text: str = ""
    format: DocumentFormat = DocumentFormat.UNKNOWN
    start_char: int | None = None
    end_char: int | None = None
    confidence: Score | None = None
    bbox: BoundingBox | None = None


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
    dimensions: Dimensions | None = None
    frame_rate: float | None = None
    format: VideoFormat = VideoFormat.UNKNOWN
