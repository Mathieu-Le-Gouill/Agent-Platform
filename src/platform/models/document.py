from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4
from enum import Enum

from models.chunk import Chunk
from models.language import Language


class DocumentType(Enum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"
    OFFICE = "office"   # docx, xlsx, pptx
    TXT = "txt"
    CSV = "csv"
    JSON = "json"
    CODE = "code"
    IMAGE = "image"   # jpg, png, webp
    AUDIO = "audio"   # mp3, wav, flac, m4a
    VIDEO = "video"   # mp4, mov
    UNKNOWN = "unknown"

    @classmethod
    def from_extension(cls, ext: str) -> "DocumentType":
        return {
            "pdf":  cls.PDF,    "md":   cls.MARKDOWN,
            "html": cls.HTML,   "htm":  cls.HTML,
            "docx": cls.OFFICE, "xlsx": cls.OFFICE,
            "pptx": cls.OFFICE, "txt":  cls.TXT,
            "csv":  cls.CSV,    "json": cls.JSON,
            "jpg":  cls.IMAGE,  "jpeg": cls.IMAGE,
            "png":  cls.IMAGE,  "webp": cls.IMAGE,
            "mp3":  cls.AUDIO,  "wav":  cls.AUDIO,
            "flac": cls.AUDIO,  "m4a":  cls.AUDIO,
            "mp4":  cls.VIDEO,  "mov":  cls.VIDEO,
        }.get(ext.lstrip(".").lower(), cls.UNKNOWN)
    
    @classmethod
    def from_mime(cls, mime: str) -> "DocumentType":
        mime = mime.split(";")[0].strip().lower()

        # Prefix-based resolution — covers every image/audio/video subtype
        if mime.startswith("image/"):  
            return cls.IMAGE
        if mime.startswith("audio/"):  
            return cls.AUDIO
        if mime.startswith("video/"):  
            return cls.VIDEO

        # Explicit table only for ambiguous text/* and application/* types
        return {
            "application/pdf":    cls.PDF,
            "text/html":          cls.HTML,
            "text/markdown":      cls.MARKDOWN,
            "text/x-markdown":    cls.MARKDOWN,
            "text/plain":         cls.TXT,
            "text/csv":           cls.CSV,
            "application/json":   cls.JSON,
            "application/msword": cls.OFFICE,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document":   cls.OFFICE,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":         cls.OFFICE,
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": cls.OFFICE,
        }.get(mime, cls.UNKNOWN)


@dataclass(slots=True, frozen=True)
class DocumentMetadata:
    author:      str | None = None
    description: str | None = None
    created_at:  datetime | None = None
    modified_at: datetime | None = None
    image: ImageMetadata | None = None
    audio: AudioMetadata | None = None
    extra:       dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ImageMetadata:
    width:       int | None = None
    height:      int | None = None
    color_space: str | None = None   # RGB, RGBA, L
    format:      str | None = None   # PNG, JPEG
    has_alpha:   bool = False


@dataclass(slots=True, frozen=True)
class AudioMetadata:
    duration_seconds: float | None = None
    sample_rate:      int | None = None   # 44100, 16000
    channels:         int | None = None   # 1=mono, 2=stereo
    format:           str | None = None   # WAV, MP3, FLAC
    bitrate:          int | None = None

   
@dataclass(slots=True)
class Document:
    id: UUID = field(default_factory=uuid4)

    # Source
    source: str = ""
    title: str | None = None
    document_type: DocumentType | None = None
    url: str | None = None

    # Content
    text: str = ""

    # Global information
    language: Language | None = None
    tags: list[str] = field(default_factory=list)

    # Lifecycle
    created_at: datetime = field(default_factory=datetime.now)

    # Arbitrary source metadata
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)

    # Chunks
    chunks: list[Chunk] = field(default_factory=list)