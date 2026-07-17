from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

pytest.importorskip("langchain_community")

from langchain_core.documents import Document as LCDocument

from agent_platform.integrations.loader.strategies.unstructured.unstructured import (
    _extract_format,
    _extract_title,
    _text_from_langchain,
    UnstructuredBaseLoader,
)
from agent_platform.core.schemas.enums import DocumentFormat, Language


class TestExtractFormat:
    def test_filetype_from_meta(self):
        meta = {"filetype": "application/pdf"}
        assert _extract_format(meta, "doc.pdf") == DocumentFormat.PDF

    def test_category_from_meta(self):
        meta = {"category": "application/pdf"}
        assert _extract_format(meta, "doc.pdf") == DocumentFormat.PDF

    def test_format_from_meta(self):
        meta = {"format": "application/pdf"}
        assert _extract_format(meta, "doc.pdf") == DocumentFormat.PDF

    def test_mime_from_meta(self):
        meta = {"mime_type": "text/plain"}
        assert _extract_format(meta, "doc.txt") == DocumentFormat.TXT

    def test_unknown_mime_returns_unknown(self):
        meta = {"filetype": "application/octet-stream"}
        assert _extract_format(meta, "doc.bin") == DocumentFormat.UNKNOWN

    def test_empty_meta_falls_back_to_extension(self):
        assert _extract_format({}, "report.pdf") == DocumentFormat.PDF
        assert _extract_format({}, "notes.markdown") == DocumentFormat.MARKDOWN
        assert _extract_format({}, "index.html") == DocumentFormat.HTML
        assert _extract_format({}, "doc.txt") == DocumentFormat.TXT

    def test_empty_meta_unknown_extension(self):
        assert _extract_format({}, "file.xyz") == DocumentFormat.UNKNOWN

    def test_no_meta_no_source_returns_unknown(self):
        assert _extract_format({}, "") == DocumentFormat.UNKNOWN

    def test_meta_takes_precedence_over_extension(self):
        meta = {"filetype": "text/html"}
        assert _extract_format(meta, "doc.pdf") == DocumentFormat.HTML

    def test_lowercase_handling(self):
        meta = {"filetype": "APPLICATION/PDF"}
        assert _extract_format(meta, "doc.PDF") == DocumentFormat.PDF


class TestExtractTitle:
    def test_title_from_meta(self):
        assert _extract_title({"title": "My Doc"}, "/path/doc.pdf") == "My Doc"

    def test_filename_from_meta(self):
        assert (
            _extract_title({"filename": "report_v2"}, "/tmp/report.pdf") == "report_v2"
        )

    def test_title_takes_precedence_over_filename(self):
        meta = {"title": "Title", "filename": "File"}
        assert _extract_title(meta, "doc.pdf") == "Title"

    def test_stem_from_source(self):
        assert _extract_title({}, "/data/report.pdf") == "report"

    def test_source_without_suffix(self):
        assert _extract_title({}, "/data/mydoc") == "mydoc"

    def test_no_metadata_no_source_returns_none(self):
        assert _extract_title({}, "") is None

    def test_empty_title_uses_filename(self):
        meta = {"title": "", "filename": "notes"}
        assert _extract_title(meta, "/path/doc.txt") == "notes"

    def test_none_title_uses_filename(self):
        meta = {"title": None, "filename": "notes"}
        assert _extract_title(meta, "path/doc.txt") == "notes"


class TestTextFromLangchain:
    def test_full_conversion_with_all_metadata(self):
        doc_id = uuid4()
        doc = LCDocument(
            page_content="Hello world\nLine two.",
            metadata={
                "source": "/path/to/doc.txt",
                "document_id": doc_id,
                "title": "My Document",
                "author": "Jane Doe",
                "created": datetime(2024, 3, 15, 10, 30),
                "last_modified": datetime(2024, 4, 1, 14, 0),
                "languages": ["en"],
                "encoding": "utf-8",
                "page_number": 3,
                "filetype": "text/plain",
            },
        )

        result = _text_from_langchain(doc)

        assert result.id == doc_id
        assert result.source == "/path/to/doc.txt"
        assert result.metadata.title == "My Document"
        assert result.metadata.author == "Jane Doe"
        assert result.metadata.created_at == datetime(2024, 3, 15, 10, 30)
        assert result.metadata.modified_at == datetime(2024, 4, 1, 14, 0)
        assert result.language == Language.EN
        assert result.encoding == "utf-8"
        assert result.format == DocumentFormat.TXT
        assert result.text == "Hello world\nLine two."
        assert result.character_count == 21
        assert result.word_count == 4
        assert result.line_count == 2
        assert result.page_count == 3

    def test_default_document_id_when_not_in_metadata(self):
        doc = LCDocument(page_content="Test", metadata={"source": "/f.txt"})
        result = _text_from_langchain(doc)
        assert result.id is not None

    def test_no_language_when_metadata_missing(self):
        doc = LCDocument(page_content="Test", metadata={"source": "/f.txt"})
        result = _text_from_langchain(doc)
        assert result.language is None


class _ConcreteLoader(UnstructuredBaseLoader):
    def _loader(self, source, config=None):
        return MagicMock()


class TestLoad:
    async def test_load_with_mocked_to_thread(self):
        mock_doc = LCDocument(
            page_content="Test content",
            metadata={"source": "/test.txt", "filetype": "text/plain"},
        )
        loader = _ConcreteLoader()

        with patch(
            "agent_platform.integrations.loader.strategies.unstructured.unstructured.asyncio.to_thread",
            return_value=[mock_doc],
        ):
            results = await loader.load("test.txt")

        assert len(results) == 1
        assert results[0].text == "Test content"
        assert results[0].source == "/test.txt"
        assert results[0].format == DocumentFormat.TXT

    async def test_load_empty_docs_returns_empty_list(self):
        loader = _ConcreteLoader()

        with patch(
            "agent_platform.integrations.loader.strategies.unstructured.unstructured.asyncio.to_thread",
            return_value=[],
        ):
            results = await loader.load("test.txt")

        assert results == []


class TestLoadMany:
    async def test_load_many_yields_results(self):
        loader = _ConcreteLoader()

        mock_doc1 = LCDocument(
            page_content="Doc1 content",
            metadata={"source": "/a.txt", "filetype": "text/plain"},
        )
        mock_doc2 = LCDocument(
            page_content="Doc2 content",
            metadata={"source": "/b.txt", "filetype": "text/plain"},
        )

        async def fake_load(source: str, config=None):
            return [_text_from_langchain(mock_doc1 if "a" in source else mock_doc2)]

        with patch.object(loader, "load", side_effect=fake_load):
            results = [docs async for docs in loader.load_many(["/a.txt", "/b.txt"])]

        assert len(results) == 2
        assert results[0][0].text == "Doc1 content"
        assert results[1][0].text == "Doc2 content"

    async def test_load_many_skips_exceptions(self):
        loader = _ConcreteLoader()

        mock_doc = LCDocument(
            page_content="Doc content",
            metadata={"source": "/a.txt", "filetype": "text/plain"},
        )

        async def fake_load(source: str, config=None):
            if "fail" in source:
                raise ValueError("Failed")
            return [_text_from_langchain(mock_doc)]

        with patch.object(loader, "load", side_effect=fake_load):
            results = [docs async for docs in loader.load_many(["/a.txt", "/fail.txt"])]

        assert len(results) == 1
        assert results[0][0].text == "Doc content"
