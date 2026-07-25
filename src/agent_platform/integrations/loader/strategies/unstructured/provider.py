import asyncio
import warnings
from abc import abstractmethod
from pathlib import Path
from uuid import uuid4

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="`langchain-community` is being sunset")
    from langchain_community.document_loaders import (
        UnstructuredFileLoader as LC_FallbackLoader,
    )
    from langchain_community.document_loaders import (
        UnstructuredHTMLLoader as LC_HTMLLoader,
    )
    from langchain_community.document_loaders import (
        UnstructuredMarkdownLoader as LC_MarkdownLoader,
    )
    from langchain_community.document_loaders import (
        UnstructuredPDFLoader as LC_PDFLoader,
    )
    from langchain_community.document_loaders import (
        UnstructuredWordDocumentLoader as LC_OfficeLoader,
    )
    from langchain_community.document_loaders.unstructured import (
        UnstructuredFileLoader as LC_UnstructuredFileLoader,
    )
from langchain_core.documents import Document as LCDocument

from agent_platform.core.interfaces.loader.text.base import BaseTextLoader
from agent_platform.core.schemas.document import DocumentMetadata, TextDocument
from agent_platform.core.schemas.enums import DocumentFormat, FileFormat, Language
from agent_platform.integrations.loader.strategies.unstructured.config import (
    UnstructuredLoaderConfig,
)


class UnstructuredBaseLoader(BaseTextLoader[UnstructuredLoaderConfig]):
    def _default_config(self) -> UnstructuredLoaderConfig:
        return UnstructuredLoaderConfig()

    @abstractmethod
    def _loader(
        self, source: str, config: UnstructuredLoaderConfig
    ) -> LC_UnstructuredFileLoader: ...

    async def load(
        self, source: str, config: UnstructuredLoaderConfig | None = None
    ) -> list[TextDocument]:
        resolved = config if config is not None else self._default_config()
        docs: list[LCDocument] = await asyncio.to_thread(
            self._loader(source, resolved).load
        )
        if not docs:
            return []
        return [_text_from_langchain(doc) for doc in docs]


class UnstructuredFileLoader(UnstructuredBaseLoader):
    def _default_config(self) -> UnstructuredLoaderConfig:
        return UnstructuredLoaderConfig()

    def _loader(
        self, source: str, config: UnstructuredLoaderConfig
    ) -> LC_UnstructuredFileLoader:
        doc_type = FileFormat.from_path(source)
        lc_cls = _LC_LOADER_MAP.get(doc_type, LC_FallbackLoader)
        kwargs = {"mode": config.mode}
        if config.chunking_strategy:
            kwargs["chunking_strategy"] = config.chunking_strategy
        return lc_cls(source, **kwargs)


_LC_LOADER_MAP: dict[FileFormat, type[LC_UnstructuredFileLoader]] = {
    FileFormat.PDF: LC_PDFLoader,
    FileFormat.MARKDOWN: LC_MarkdownLoader,
    FileFormat.HTML: LC_HTMLLoader,
    FileFormat.DOCX: LC_OfficeLoader,
}


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
