from abc import abstractmethod
from typing import AsyncIterator, Sequence
import asyncio
from pathlib import Path
from uuid import uuid4
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="`langchain-community` is being sunset")
    from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_core.documents import Document as LCDocument

from agent_platform.integrations.loader.text_base import BaseTextLoader
from agent_platform.models.document import TextDocument, DocumentMetadata
from agent_platform.models.enums import DocumentFormat, Language, FileFormat


class UnstructuredBaseLoader(BaseTextLoader):
    @abstractmethod
    def _loader(self, source: str, **kwargs) -> UnstructuredFileLoader: ...

    async def load(self, source: str, **kwargs) -> Sequence[TextDocument]:
        docs: list[LCDocument] = await asyncio.to_thread(
            self._loader(source, **kwargs).load
        )
        if not docs:
            return []
        return [_text_from_langchain(doc) for doc in docs]

    async def load_many(
        self, sources: list[str], **kwargs
    ) -> AsyncIterator[Sequence[TextDocument]]:
        results = await asyncio.gather(
            *[self.load(s, **kwargs) for s in sources],
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                continue
            yield result


def _text_from_langchain(doc: LCDocument) -> TextDocument:
    metadata = doc.metadata
    text = doc.page_content
    source = metadata.get("source") or ""
    fmt = _extract_format(metadata, source)

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
        format=fmt,
        encoding=metadata.get("encoding"),
        language=(
            Language(metadata["languages"][0]) if metadata.get("languages") else None
        ),
        character_count=len(text),
        word_count=len(text.split()),
        line_count=text.count("\n") + 1,
        page_count=metadata.get("page_number"),
    )


def _extract_format(meta: dict, source: str = "") -> DocumentFormat:
    raw = (
        meta.get("filetype")
        or meta.get("category")
        or meta.get("format")
        or meta.get("mime_type")
    )
    if raw:
        ff = FileFormat.from_mime(raw.lower())
        return (
            ff.extension_format
            if isinstance(ff.extension_format, DocumentFormat)
            else DocumentFormat.UNKNOWN
        )
    ext = Path(source).suffix if source else ""
    if ext:
        ff = FileFormat.from_extension(ext.lstrip("."))
        return (
            ff.extension_format
            if isinstance(ff.extension_format, DocumentFormat)
            else DocumentFormat.UNKNOWN
        )
    return DocumentFormat.UNKNOWN


def _extract_title(meta: dict, source: str) -> str | None:
    return (
        meta.get("title")
        or meta.get("filename")
        or (Path(source).stem if source else None)
        or None
    )
