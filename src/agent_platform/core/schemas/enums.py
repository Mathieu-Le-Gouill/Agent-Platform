from __future__ import annotations

from enum import Enum
from typing import Union
from pathlib import Path


class MediaType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    UNKNOWN = "unknown"


class DocumentFormat(str, Enum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"
    LATEX = "latex"
    TXT = "txt"
    CSV = "csv"
    JSON = "json"
    DOCX = "docx"
    XLSX = "xlsx"
    PPTX = "pptx"
    UNKNOWN = "unknown"


class ImageFormat(str, Enum):
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    GIF = "gif"
    BMP = "bmp"
    TIFF = "tiff"
    UNKNOWN = "unknown"


class AudioFormat(str, Enum):
    MP3 = "mp3"
    WAV = "wav"
    FLAC = "flac"
    AAC = "aac"
    M4A = "m4a"
    OGG = "ogg"
    UNKNOWN = "unknown"


class VideoFormat(str, Enum):
    MP4 = "mp4"
    MOV = "mov"
    AVI = "avi"
    MKV = "mkv"
    WEBM = "webm"
    UNKNOWN = "unknown"


class DataType(str, Enum):
    INT16 = "int16"
    FLOAT32 = "float32"
    INT8 = "int8"
    UINT8 = "uint8"


class Language(str, Enum):
    AF = "af"
    AM = "am"
    AR = "ar"
    AS = "as"
    AZ = "az"
    EN = "en"
    FR = "fr"
    GE = "de"
    SP = "es"
    CH = "zh"
    JA = "ja"
    KO = "ko"
    IT = "it"
    PO = "pt"
    RU = "ru"

    @property
    def code(self) -> str:
        return self.value


MediaFormat = Union[DocumentFormat, ImageFormat, AudioFormat, VideoFormat]


_MIME_TO_FORMAT: dict[str, "FileFormat"] = {}
_FORMAT_BY_EXTENSION: dict[str, "FileFormat"] = {}


class FileFormat(Enum):
    # ----- Text -----
    PDF = (DocumentFormat.PDF, MediaType.TEXT)
    MARKDOWN = (DocumentFormat.MARKDOWN, MediaType.TEXT)
    HTML = (DocumentFormat.HTML, MediaType.TEXT)
    LATEX = (DocumentFormat.LATEX, MediaType.TEXT)
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
        return self.extension_format.value if self.extension_format else None

    @property
    def media(self):
        return self.media_type.value

    @classmethod
    def from_extension(cls, ext: str) -> FileFormat:
        ext = ext.removeprefix(".").lower()
        return _FORMAT_BY_EXTENSION.get(ext, cls.UNKNOWN)

    @classmethod
    def from_path(cls, path: str) -> FileFormat:
        return cls.from_extension(Path(path).suffix)

    @classmethod
    def from_mime(cls, mime: str) -> FileFormat:
        mime = mime.split(";")[0].strip().lower()
        if mime in _MIME_TO_FORMAT:
            return _MIME_TO_FORMAT[mime]
        if mime.startswith(("image/", "audio/", "video/")):
            return cls.UNKNOWN
        return cls.UNKNOWN

    @classmethod
    def from_media_type(cls, media_type: MediaType, fmt: str) -> FileFormat:
        for f in cls:
            if f.media_type == media_type and f.extension == fmt:
                return f
        return cls.UNKNOWN


# --- Lookup tables ---

_MIME_TO_FORMAT.update(
    {
        # Documents
        "application/pdf": FileFormat.PDF,
        "text/plain": FileFormat.TXT,
        "text/html": FileFormat.HTML,
        "text/markdown": FileFormat.MARKDOWN,
        "text/x-markdown": FileFormat.MARKDOWN,
        "application/x-latex": FileFormat.LATEX,
        "application/x-tex": FileFormat.LATEX,
        "text/x-tex": FileFormat.LATEX,
        "text/csv": FileFormat.CSV,
        "application/json": FileFormat.JSON,
        "application/msword": FileFormat.DOCX,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": FileFormat.DOCX,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": FileFormat.XLSX,
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": FileFormat.PPTX,
        # Images
        "image/jpeg": FileFormat.JPEG,
        "image/png": FileFormat.PNG,
        "image/webp": FileFormat.WEBP,
        "image/gif": FileFormat.GIF,
        "image/bmp": FileFormat.BMP,
        "image/tiff": FileFormat.TIFF,
        # Audio
        "audio/mpeg": FileFormat.MP3,
        "audio/wav": FileFormat.WAV,
        "audio/x-wav": FileFormat.WAV,
        "audio/flac": FileFormat.FLAC,
        "audio/aac": FileFormat.AAC,
        "audio/mp4": FileFormat.M4A,
        "audio/ogg": FileFormat.OGG,
        "audio/vorbis": FileFormat.OGG,
        # Video
        "video/mp4": FileFormat.MP4,
        "video/quicktime": FileFormat.MOV,
        "video/x-msvideo": FileFormat.AVI,
        "video/x-matroska": FileFormat.MKV,
        "video/webm": FileFormat.WEBM,
    }
)

_FORMAT_BY_EXTENSION.update(
    {
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
        "tex": FileFormat.LATEX,
        "latex": FileFormat.LATEX,
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
)


class SimilarityMetric(str, Enum):
    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


class FinishReason(str, Enum):
    STOP = "stop"
    STOP_SEQUENCE = "stop_sequence"
    LENGTH = "length"
    TOOL_CALL = "tool_call"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
    UNKNOWN = "unknown"
