from __future__ import annotations
from enum import Enum


class MediaType(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    UNKNOWN = "unknown"


class FileFormat(Enum):
    # ----- Text -----
    PDF = ("pdf", MediaType.TEXT)
    MARKDOWN = ("markdown", MediaType.TEXT)
    HTML = ("html", MediaType.TEXT)
    TXT = ("txt", MediaType.TEXT)
    CSV = ("csv", MediaType.TEXT)
    JSON = ("json", MediaType.TEXT)
    DOCX = ("docx", MediaType.TEXT)
    XLSX = ("xlsx", MediaType.TEXT)
    PPTX = ("pptx", MediaType.TEXT)

    # ----- Images -----
    JPEG = ("jpeg", MediaType.IMAGE)
    PNG = ("png", MediaType.IMAGE)
    WEBP = ("webp", MediaType.IMAGE)
    GIF = ("gif", MediaType.IMAGE)
    BMP = ("bmp", MediaType.IMAGE)
    TIFF = ("tiff", MediaType.IMAGE)

    # ----- Audio -----
    MP3 = ("mp3", MediaType.AUDIO)
    WAV = ("wav", MediaType.AUDIO)
    FLAC = ("flac", MediaType.AUDIO)
    AAC = ("aac", MediaType.AUDIO)
    M4A = ("m4a", MediaType.AUDIO)
    OGG = ("ogg", MediaType.AUDIO)

    # ----- Video -----
    MP4 = ("mp4", MediaType.VIDEO)
    MOV = ("mov", MediaType.VIDEO)
    AVI = ("avi", MediaType.VIDEO)
    MKV = ("mkv", MediaType.VIDEO)
    WEBM = ("webm", MediaType.VIDEO)

    UNKNOWN = ("unknown", MediaType.UNKNOWN)


    def __init__(self, extension: str, media_type: MediaType):
        self.extension = extension
        self.media_type = media_type
    
    
    @classmethod
    def from_extension(cls, extension: str) -> "FileFormat":
        extension = extension.removeprefix(".").lower()
        return _EXTENSION_TO_FORMAT.get(extension, cls.UNKNOWN)

    @classmethod
    def from_mime(cls, mime: str) -> "FileFormat":
        mime = mime.split(";")[0].strip().lower()

        # Generic MIME prefixes
        if mime.startswith("image/"):
            subtype = mime.split("/", 1)[1]
            return _EXTENSION_TO_FORMAT.get(subtype, cls.UNKNOWN)

        if mime.startswith("audio/"):
            subtype = mime.split("/", 1)[1]
            return _EXTENSION_TO_FORMAT.get(subtype, cls.UNKNOWN)

        if mime.startswith("video/"):
            subtype = mime.split("/", 1)[1]
            return _EXTENSION_TO_FORMAT.get(subtype, cls.UNKNOWN)

        return _MIME_TO_FORMAT.get(mime, cls.UNKNOWN)
    

_EXTENSION_TO_FORMAT = {
    fmt.extension: fmt
    for fmt in FileFormat
    if fmt.extension
}


# Common aliases
_EXTENSION_TO_FORMAT.update({
    "jpg": FileFormat.JPEG,
    "htm": FileFormat.HTML,
})


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