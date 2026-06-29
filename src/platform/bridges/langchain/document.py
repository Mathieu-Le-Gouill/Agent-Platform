from pathlib import Path
from datetime import datetime
from typing import Sequence

from models.document import Document, DocumentMetadata, DocumentType, ImageMetadata, AudioMetadata
from models.language import Language

from langchain_core.documents import Document as LCDocument
from bridges.langchain.chunk import from_langchain as chunk_from_lc, to_langchain as chunk_to_lc


_EMAIL_FIELDS = ("sent_from", "sent_to", "cc_recipient", "bcc_recipient",
                 "email_message_id", "subject", "signature")


def to_langchain(doc: Document) -> list[LCDocument]:
    base_metadata = {
        "document_id":   str(doc.id),
        "source":        doc.source,
        "title":         doc.title,
        "document_type": doc.document_type.value if doc.document_type else None,
        "url":           doc.url,
        "language":      str(doc.language) if doc.language else None,
        "tags":          doc.tags,
        "created_at":    doc.created_at.isoformat(),
        "author":        doc.metadata.author,
        "description":   doc.metadata.description,
        **doc.metadata.extra,
    }

    if doc.chunks:
        lc_docs = []
        for chunk in doc.chunks:
            lc_doc = chunk_to_lc(chunk)
            lc_doc.metadata = {**base_metadata, **lc_doc.metadata}
            lc_docs.append(lc_doc)
        return lc_docs

    return [LCDocument(id=str(doc.id), page_content=doc.text or "", metadata=base_metadata)]


def from_langchain(lc_docs: Sequence[LCDocument], source: str = "") -> Document:
    first_meta = lc_docs[0].metadata
    all_meta   = [d.metadata for d in lc_docs]
    doc_type   = _extract_document_type(first_meta, source)

    return Document(
        source        = source,
        title         = _extract_title(lc_docs, first_meta, source),
        document_type = doc_type,
        url           = first_meta.get("url") or _data_source(first_meta).get("url"),
        text          = "\n\n".join(d.page_content for d in lc_docs),
        language      = _parse_language(first_meta),
        tags          = _extract_tags(first_meta),
        created_at    = _parse_datetime(first_meta, "date_created", "creation_date") or datetime.now(),
        metadata=DocumentMetadata(
            author      = first_meta.get("author") or None,
            description = first_meta.get("description") or None,
            created_at  = _parse_datetime(first_meta, "date_created", "creation_date"),
            modified_at = _parse_datetime(first_meta, "last_modified", "date_modified"),
            image       = _extract_image_metadata(first_meta) if doc_type == DocumentType.IMAGE else None,
            audio       = _extract_audio_metadata(first_meta) if doc_type == DocumentType.AUDIO else None,
            extra={
                "page_count":    _count_pages(lc_docs),
                "file_metadata": first_meta,
                "page_metadata": all_meta,
                **{k: v for k in _EMAIL_FIELDS if (v := first_meta.get(k))},
            },
        ),
        chunks=[chunk_from_lc(doc) for doc in lc_docs],
    )


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
    """Try each key in order, also checking data_source; return the first parseable value."""
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