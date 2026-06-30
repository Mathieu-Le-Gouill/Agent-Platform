from __future__ import annotations
from enum import Enum


class FileFormat(Enum):
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
    def from_extension(cls, ext: str) -> FileFormat:
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
    def from_mime(cls, mime: str) -> FileFormat:
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