import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="`langchain-community` is being sunset")
    from langchain_community.document_loaders import (
        UnstructuredPDFLoader as LC_PDFLoader,
        UnstructuredMarkdownLoader as LC_MarkdownLoader,
        UnstructuredHTMLLoader as LC_HTMLLoader,
        UnstructuredWordDocumentLoader as LC_OfficeLoader,
        UnstructuredFileLoader as LC_FallbackLoader,
    )
    from langchain_community.document_loaders.unstructured import (
        UnstructuredFileLoader as LC_UnstructuredFileLoader,
    )
from agent_platform.integrations.loader.unstructured_base import UnstructuredBaseLoader
from agent_platform.models.enums import FileFormat


class UnstructuredFileLoader(UnstructuredBaseLoader):
    def _loader(self, source: str, **kwargs) -> LC_UnstructuredFileLoader:

        doc_type = FileFormat.from_extension(source)
        lc_cls = _LC_LOADER_MAP.get(doc_type, LC_FallbackLoader)

        kwargs.setdefault("mode", "elements")

        return lc_cls(source, **kwargs)


_LC_LOADER_MAP: dict[FileFormat, type[LC_UnstructuredFileLoader]] = {
    FileFormat.PDF: LC_PDFLoader,
    FileFormat.MARKDOWN: LC_MarkdownLoader,
    FileFormat.HTML: LC_HTMLLoader,
    FileFormat.DOCX: LC_OfficeLoader,
}
