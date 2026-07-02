from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union


class MediaType(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    UNKNOWN = "unknown"

    @classmethod
    def from_fileformat(cls, fileformat: FileFormat) -> MediaType:
        return fileformat.media_type if isinstance(fileformat, FileFormat) else cls.UNKNOWN


class DocumentFormat(Enum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"
    TXT = "txt"
    CSV = "csv"
    JSON = "json"
    DOCX = "docx"
    XLSX = "xlsx"
    PPTX = "pptx"
    UNKNOWN = "unknown"

    @classmethod
    def from_fileformat(cls, fileformat: FileFormat) -> DocumentFormat:
        return fileformat.extension_format if isinstance(fileformat.extension_format, DocumentFormat) else cls.UNKNOWN


class ImageFormat(Enum):
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    GIF = "gif"
    BMP = "bmp"
    TIFF = "tiff"
    UNKNOWN = "unknown"

    @classmethod
    def from_fileformat(cls, fileformat: FileFormat) -> ImageFormat:
        return fileformat.extension_format if isinstance(fileformat.extension_format, ImageFormat) else cls.UNKNOWN


class AudioFormat(Enum):
    MP3 = "mp3"
    WAV = "wav"
    FLAC = "flac"
    AAC = "aac"
    M4A = "m4a"
    OGG = "ogg"
    UNKNOWN = "unknown"

    @classmethod
    def from_fileformat(cls, fileformat: FileFormat) -> AudioFormat:
        return fileformat.extension_format if isinstance(fileformat.extension_format, AudioFormat) else cls.UNKNOWN


class VideoFormat(Enum):
    MP4 = "mp4"
    MOV = "mov"
    AVI = "avi"
    MKV = "mkv"
    WEBM = "webm"
    UNKNOWN = "unknown"

    @classmethod
    def from_fileformat(cls, fileformat: FileFormat) -> VideoFormat:
        return fileformat.extension_format if isinstance(fileformat.extension_format, VideoFormat) else cls.UNKNOWN


MediaFormat = Union[
    DocumentFormat,
    ImageFormat,
    AudioFormat,
    VideoFormat,
]


class FileFormat(Enum):
    # ----- Text -----
    PDF = (DocumentFormat.PDF, MediaType.TEXT)
    MARKDOWN = (DocumentFormat.MARKDOWN, MediaType.TEXT)
    HTML = (DocumentFormat.HTML, MediaType.TEXT)
    TXT = (DocumentFormat.TXT, MediaType.TEXT)
    CSV = (DocumentFormat.CSV, MediaType.TEXT)
    JSON = (DocumentFormat.JSON, MediaType.TEXT)
    DOCX = (DocumentFormat.DOCX, MediaType.TEXT)
    XLSX = (DocumentFormat.XLSX, MediaType.TEXT)
    PPTX = (DocumentFormat.PPTX, MediaType.TEXT)

    # ----- Images -----
    JPEG = (ImageFormat.JPEG, MediaType.IMAGE)
    PNG = (ImageFormat.PNG, MediaType.IMAGE)
    WEBP = (ImageFormat.WEBP, MediaType.IMAGE)
    GIF = (ImageFormat.GIF, MediaType.IMAGE)
    BMP = (ImageFormat.BMP, MediaType.IMAGE)
    TIFF = (ImageFormat.TIFF, MediaType.IMAGE)

    # ----- Audio -----
    MP3 = (AudioFormat.MP3, MediaType.AUDIO)
    WAV = (AudioFormat.WAV, MediaType.AUDIO)
    FLAC = (AudioFormat.FLAC, MediaType.AUDIO)
    AAC = (AudioFormat.AAC, MediaType.AUDIO)
    M4A = (AudioFormat.M4A, MediaType.AUDIO)
    OGG = (AudioFormat.OGG, MediaType.AUDIO)

    # ----- Video -----
    MP4 = (VideoFormat.MP4, MediaType.VIDEO)
    MOV = (VideoFormat.MOV, MediaType.VIDEO)
    AVI = (VideoFormat.AVI, MediaType.VIDEO)
    MKV = (VideoFormat.MKV, MediaType.VIDEO)
    WEBM = (VideoFormat.WEBM, MediaType.VIDEO)

    UNKNOWN = (None, MediaType.UNKNOWN)


    def __init__(self, extension_format: MediaFormat | None, media_type: MediaType):
        self.extension_format = extension_format
        self.media_type = media_type


    @property
    def extension(self):
        if self.extension_format is None:
            return None
        return self.extension_format.value
    

    @property
    def media(self):
        return self.media_type.value
    

    @classmethod
    def from_extension(cls, ext: str) -> FileFormat:
        return FORMAT_BY_EXTENSION.get(ext.lower()) or cls.UNKNOWN


    @classmethod
    def from_mime(cls, mime: str) -> FileFormat:
        mime = mime.split(";")[0].strip().lower()

        # direct mapping first
        if mime in _MIME_TO_FORMAT:
            return _MIME_TO_FORMAT[mime]

        # fallback prefix logic
        if mime.startswith("image/"):
            return cls.UNKNOWN
        if mime.startswith("audio/"):
            return cls.UNKNOWN
        if mime.startswith("video/"):
            return cls.UNKNOWN

        return cls.UNKNOWN


_MIME_TO_FORMAT = {
    # Text
    "application/pdf": FileFormat.PDF,
    "text/plain": FileFormat.TXT,
    "text/html": FileFormat.HTML,
    "text/markdown": FileFormat.MARKDOWN,
    "text/x-markdown": FileFormat.MARKDOWN,
    "text/csv": FileFormat.CSV,
    "application/json": FileFormat.JSON,

    # Office
    "application/msword": FileFormat.DOCX,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": FileFormat.DOCX,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": FileFormat.XLSX,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": FileFormat.PPTX,
}



FORMAT_BY_EXTENSION: dict[str, FileFormat] = {
    # Documents
    "pdf": FileFormat.PDF,
    "docx": FileFormat.DOCX,
    "xlsx": FileFormat.XLSX,
    "pptx": FileFormat.PPTX,
    "txt": FileFormat.TXT,
    "csv": FileFormat.CSV,
    "json": FileFormat.JSON,
    "html": FileFormat.HTML,
    "markdown": FileFormat.MARKDOWN,

    # Images
    "png": FileFormat.PNG,
    "jpg": FileFormat.JPEG,
    "jpeg": FileFormat.JPEG,
    "webp": FileFormat.WEBP,
    "gif": FileFormat.GIF,
    "bmp": FileFormat.BMP,
    "tiff": FileFormat.TIFF,

    # Audio
    "mp3": FileFormat.MP3,
    "wav": FileFormat.WAV,
    "flac": FileFormat.FLAC,
    "aac": FileFormat.AAC,
    "m4a": FileFormat.M4A,
    "ogg": FileFormat.OGG,

    # Video
    "mp4": FileFormat.MP4,
    "mov": FileFormat.MOV,
    "avi": FileFormat.AVI,
    "mkv": FileFormat.MKV,
    "webm": FileFormat.WEBM,
}