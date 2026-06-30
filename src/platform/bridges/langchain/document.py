from pathlib import Path
from datetime import datetime
from typing import Sequence

from models.document import TextDocument
from platform.models.enums.language import Language

from langchain_core.documents import Document as LC_Document


_EMAIL_FIELDS = ("sent_from", "sent_to", "cc_recipient", "bcc_recipient",
                 "email_message_id", "subject", "signature")


def to_langchain(doc: TextDocument) -> LC_Document:
    
    return LC_Document(
        page_content=doc.text,
        metadata={
            "document_id": str(doc.id),
            "source": doc.source,

            "title": doc.metadata.title,
            "author": doc.metadata.author,
            "description": doc.metadata.description,
            "format": (
                doc.metadata.format.value
                if doc.metadata.format
                else None
            ),
            "created_at": doc.metadata.created_at,
            "modified_at": doc.metadata.modified_at,
            "extra": doc.metadata.extra,
            
            "mime_type": doc.info.mime_type,
            "encoding": doc.info.encoding,
            "language": (
                doc.info.language.code
                if doc.info.language
                else None
            ),
        },
    )

"""
# --- helpers ---

def _data_source(meta: dict) -> dict:
    ds = meta.get("data_source") or {}
    return ds if isinstance(ds, dict) else {}


def _parse_language(meta: dict) -> Language | None:
    raw = meta.get("language") or next(iter(meta.get("languages") or []), None)
    try:
        return Language(raw) if raw else None
    except ValueError:
        return None


def _parse_datetime(meta: dict, *keys: str) -> datetime | None:
    for key in keys:
        raw = meta.get(key) or _data_source(meta).get(key)
        if raw:
            try:
                return datetime.fromisoformat(str(raw))
            except ValueError:
                continue
    return None


def _extract_document_type(meta: dict, source: str = "") -> DocumentType | None:
    raw = meta.get("filetype") or meta.get("category")
    if raw:
        return DocumentType.from_mime(raw.lower())
    ext = Path(source).suffix if source else ""
    return DocumentType.from_extension(ext) if ext else None


def _extract_title(docs: Sequence[LCDocument], first_meta: dict, source: str) -> str | None:
    titled = next((d.page_content for d in docs if d.metadata.get("category") == "Title"), None)
    return titled or first_meta.get("filename") or (Path(source).stem if source else None) or None


def _extract_tags(meta: dict) -> list[str]:
    raw = meta.get("keywords") or meta.get("subject") or ""
    if isinstance(raw, list):
        return [str(k).strip() for k in raw if str(k).strip()]
    return [kw.strip() for kw in str(raw).split(",") if kw.strip()]


def _count_pages(docs: Sequence[LCDocument]) -> int:
    page_numbers: set[int] = {
        d.metadata["page_number"]
        for d in docs
        if d.metadata.get("page_number") is not None
    }
    return max(page_numbers) if page_numbers else len(docs)


def _extract_image_metadata(meta: dict) -> ImageMetadata | None:
    if not any(meta.get(k) for k in ("image_width", "image_height", "image_mime_type", "image_format")):
        return None
    return ImageMetadata(
        width       = meta.get("image_width"),
        height      = meta.get("image_height"),
        color_space = meta.get("color_space"),
        format      = meta.get("image_format") or meta.get("image_mime_type"),
        has_alpha   = meta.get("has_alpha", False),
    )


def _extract_audio_metadata(meta: dict) -> AudioMetadata | None:
    if not any(meta.get(k) for k in ("duration", "sample_rate", "audio_format")):
        return None
    return AudioMetadata(
        duration_seconds = meta.get("duration"),
        sample_rate      = meta.get("sample_rate"),
        channels         = meta.get("channels"),
        format           = meta.get("audio_format"),
        bitrate          = meta.get("bitrate"),
    )

"""