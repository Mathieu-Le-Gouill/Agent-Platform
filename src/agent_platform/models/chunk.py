
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from agent_platform.models.enums.file_format import VideoFormat, DocumentFormat, AudioFormat
from agent_platform.models.enums.dtype import DataType


@dataclass(slots=True, frozen=True, kw_only=True)
class Chunk:
    id: UUID
    index: int | None = None
    document_id: UUID | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True, kw_only=True)
class TextChunk(Chunk):
    text: str
    format: DocumentFormat | None = None
    start_char: int | None = None
    end_char: int | None = None


@dataclass(slots=True, frozen=True, kw_only=True)
class AudioChunk(Chunk):
    data: bytes
    sample_rate: int
    start: int # start sample id
    end: int # end sample id
    dtype: DataType # int16, float32
    channels: int = 1
    format: AudioFormat | None = None # .wav, .mp3


@dataclass(slots=True, frozen=True, kw_only=True)
class VideoChunk(Chunk):
    data: bytes
    start: int
    end: int
    width: int | None = None
    height: int | None = None
    frame_rate: float | None = None
    format: VideoFormat | None = None