from langchain_community.document_loaders import (
    UnstructuredPDFLoader as LC_PDFLoader,
    UnstructuredMarkdownLoader as LC_MarkdownLoader,
    UnstructuredHTMLLoader as LC_HTMLLoader,
    UnstructuredWordDocumentLoader as LC_OfficeLoader,
    UnstructuredFileLoader as LC_FallbackLoader,
)
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from integrations.loader.base import BaseLoader
from models.document import DocumentType


_LC_LOADER_MAP: dict[DocumentType, type[UnstructuredFileLoader]] = {
    DocumentType.PDF:      LC_PDFLoader,
    DocumentType.MARKDOWN: LC_MarkdownLoader,
    DocumentType.HTML:     LC_HTMLLoader,
    DocumentType.OFFICE:   LC_OfficeLoader,
}

class DocumentLoader(BaseLoader):

    def _loader(self, source: str, **kwargs) -> UnstructuredFileLoader:

        doc_type = DocumentType.from_extension(source)
        lc_cls = _LC_LOADER_MAP.get(doc_type, LC_FallbackLoader)

        kwargs.setdefault("mode", "elements")

        return lc_cls(source, **kwargs)