from mimetypes import guess_type
from pathlib import Path
from typing import Callable, Sequence
from langchain_core.documents import Document as LCDocument

from agent_platform.models.document import (
    Document, DocumentMetadata,
    TextDocument, TextInfo, TextProperties,
    ImageDocument, AudioDocument, VideoDocument,
)
from agent_platform.models.enums.file_format import DocumentFormat, FileFormat, MediaType
from uuid import uuid4




_EMAIL_FIELDS = ("sent_from", "sent_to", "cc_recipient", "bcc_recipient",
                 "email_message_id", "subject", "signature")


def to_langchain(doc: TextDocument) -> LCDocument:
    
    return LCDocument(
        page_content=doc.text,
        metadata={
            "document_id": doc.id,
            "source": doc.source,

            "title": doc.metadata.title,
            "author": doc.metadata.author,
            "description": doc.metadata.description,
            "format": doc.info.format,

            "created_at": doc.metadata.created_at,
            "modified_at": doc.metadata.modified_at,
            "extra": doc.metadata.extra,
            
            "encoding": doc.info.encoding,
            "language": doc.info.language,
        },
    )


def from_langchain(
    lc_docs: list[LCDocument],
    source: str = "",
) -> Sequence[Document]:
    mime, _ = guess_type(source)

    file_format = (
        FileFormat.from_mime(mime)
        if mime
        else FileFormat.UNKNOWN
    )

    converter = _CONVERTERS.get(file_format.media_type)

    if converter is None:
        return []

    return [converter(doc) for doc in lc_docs]


def text_from_langchain(doc: LCDocument) -> TextDocument:
    metadata = doc.metadata

    text = doc.page_content
    source = metadata.get("source") or ""

    format = _extract_document_type(metadata, source)


    return TextDocument(
        id=metadata.get("document_id") or uuid4(),
        source=source,
        metadata=DocumentMetadata(
            title=_extract_title(metadata, source),
            author=metadata.get("author"),
            description=None,
            created_at=metadata.get("created"),
            modified_at=metadata.get("last_modified"),
            extra=metadata,
        ),
        text=text,
        info=TextInfo(
            format=DocumentFormat.from_fileformat(format),
            encoding=metadata.get("encoding"),
            language=(
                metadata["languages"][0]
                if metadata.get("languages")
                else None
            ),
        ),
        properties=TextProperties(
            character_count=len(text),
            word_count=len(text.split()),
            line_count=text.count("\n") + 1,
            page_count=metadata.get("page_number"),
        ),
    )

def image_from_langchain(doc: LCDocument) -> ImageDocument:
    ...

def audio_from_langchain(doc: LCDocument) -> AudioDocument:
    ...


def video_from_langchain(doc: LCDocument) -> VideoDocument:
    ...


_CONVERTERS: dict[MediaType, Callable[[LCDocument], Document]] = {
    MediaType.TEXT: text_from_langchain,
    MediaType.IMAGE: image_from_langchain,
    MediaType.AUDIO: audio_from_langchain,
    MediaType.VIDEO: video_from_langchain,
}


# --- helpers ---

def _extract_document_type(meta: dict, source: str = "") -> FileFormat:

    raw = meta.get("filetype") or meta.get("category") or meta.get("format") or meta.get("mime_type")
    if raw:
        return FileFormat.from_mime(raw.lower())
    ext = Path(source).suffix if source else ""
    return FileFormat.from_extension(ext) if ext else FileFormat.UNKNOWN


def _extract_title(meta: dict, source: str) -> str | None:

    return meta.get("title") or meta.get("filename") or (Path(source).stem if source else None) or None