from datetime import datetime
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

pytest.importorskip("langchain_core")

from langchain_core.documents import Document as LC_Document

from agent_platform.core.interfaces.chunking.config import ChunkerConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.document import DocumentMetadata, TextDocument
from agent_platform.core.schemas.enums import DocumentFormat, Language
from agent_platform.integrations.chunking.langchain_base import (
    LangChainChunker,
    _doc_to_lc,
    _lc_to_chunks,
)

# ================================================================
# _doc_to_lc  mapper
# ================================================================


def test_doc_to_lc_maps_all_fields():
    doc_id = uuid4()
    created = datetime(2025, 1, 1)
    modified = datetime(2025, 6, 1)
    doc = TextDocument(
        id=doc_id,
        text="Hello world",
        source="test.txt",
        format=DocumentFormat.TXT,
        language=Language.EN,
        encoding="utf-8",
        metadata=DocumentMetadata(
            title="Test",
            author="me",
            description="desc",
            created_at=created,
            modified_at=modified,
            extra={"key": "val"},
        ),
    )
    lc = _doc_to_lc(doc)
    assert isinstance(lc, LC_Document)
    assert lc.page_content == "Hello world"
    assert lc.metadata["document_id"] == str(doc_id)
    assert lc.metadata["source"] == "test.txt"
    assert lc.metadata["title"] == "Test"
    assert lc.metadata["author"] == "me"
    assert lc.metadata["description"] == "desc"
    assert lc.metadata["format"] == "txt"
    assert lc.metadata["created_at"] == created
    assert lc.metadata["modified_at"] == modified
    assert lc.metadata["extra"] == {"key": "val"}
    assert lc.metadata["encoding"] == "utf-8"
    assert lc.metadata["language"] == "en"


def test_doc_to_lc_none_language():
    doc = TextDocument(text="hello", language=None)
    lc = _doc_to_lc(doc)
    assert lc.metadata["format"] == "unknown"
    assert lc.metadata["language"] is None


def test_doc_to_lc_default_metadata():
    doc = TextDocument(text="hello")
    lc = _doc_to_lc(doc)
    assert lc.metadata["title"] is None
    assert lc.metadata["author"] is None
    assert lc.metadata["description"] is None
    assert lc.metadata["created_at"] is None
    assert lc.metadata["modified_at"] is None
    assert lc.metadata["extra"] == {}
    assert lc.metadata["encoding"] is None


# ================================================================
# _lc_to_chunks  mapper  (with edge case tests)
# ================================================================


class TestLcToChunks:
    def test_basic(self):
        lc_docs = [
            LC_Document(
                page_content="chunk one",
                metadata={"index": 0, "format": "txt"},
            ),
            LC_Document(
                page_content="chunk two",
                metadata={"index": 1, "format": "txt"},
            ),
        ]
        chunks = _lc_to_chunks(lc_docs)
        assert len(chunks) == 2
        assert all(isinstance(c, TextChunk) for c in chunks)
        assert chunks[0].text == "chunk one"
        assert chunks[0].index == 0
        assert chunks[1].text == "chunk two"
        assert chunks[1].index == 1

    def test_uses_chunk_id_from_metadata(self):
        expected = uuid4()
        lc_docs = [
            LC_Document(
                page_content="text",
                metadata={"chunk_id": str(expected), "format": "txt"},
            ),
        ]
        chunks = _lc_to_chunks(lc_docs)
        assert chunks[0].id == expected

    def test_uses_document_id_from_metadata(self):
        expected = uuid4()
        lc_docs = [
            LC_Document(
                page_content="text",
                metadata={"document_id": str(expected), "format": "txt"},
            ),
        ]
        chunks = _lc_to_chunks(lc_docs)
        assert chunks[0].document_id == expected

    def test_missing_index_defaults_to_zero(self):
        lc_docs = [LC_Document(page_content="text", metadata={"format": "txt"})]
        chunks = _lc_to_chunks(lc_docs)
        assert chunks[0].index == 0

    def test_explicit_index_zero(self):
        lc_docs = [
            LC_Document(page_content="text", metadata={"index": 0, "format": "txt"})
        ]
        chunks = _lc_to_chunks(lc_docs)
        assert chunks[0].index == 0

    def test_start_char_and_end_char(self):
        lc_docs = [
            LC_Document(
                page_content="hello there",
                metadata={"start_char": 10, "format": "txt"},
            ),
        ]
        chunks = _lc_to_chunks(lc_docs)
        assert chunks[0].start_char == 10
        assert chunks[0].end_char == 21

    def test_start_char_none_leaves_end_char_none(self):
        lc_docs = [LC_Document(page_content="text", metadata={"format": "txt"})]
        chunks = _lc_to_chunks(lc_docs)
        assert chunks[0].start_char is None
        assert chunks[0].end_char is None

    def test_format_from_metadata(self):
        lc_docs = [
            LC_Document(
                page_content="text",
                metadata={"format": "markdown", "extra": {}},
            ),
        ]
        chunks = _lc_to_chunks(lc_docs)
        assert chunks[0].format == DocumentFormat.MARKDOWN

    def test_language_from_metadata(self):
        lc_docs = [
            LC_Document(
                page_content="text",
                metadata={"format": "txt", "language": "fr", "extra": {}},
            ),
        ]
        chunks = _lc_to_chunks(lc_docs)
        assert chunks[0].metadata["language"] == Language.FR

    def test_handles_extra_metadata_fallback(self):
        lc_docs = [LC_Document(page_content="text", metadata={"format": "txt"})]
        chunks = _lc_to_chunks(lc_docs)
        assert chunks[0].metadata["extra"] == {}

    def test_uses_start_index_alias(self):
        lc_docs = [
            LC_Document(
                page_content="text",
                metadata={"start_index": 5, "format": "txt"},
            ),
        ]
        chunks = _lc_to_chunks(lc_docs)
        assert chunks[0].start_char == 5

    # --- Error / edge cases (were completely missing) ---

    def test_invalid_chunk_id_fallback(self):
        lc_docs = [
            LC_Document(
                page_content="text",
                metadata={"chunk_id": "not-a-valid-uuid", "format": "txt"},
            ),
        ]
        chunks = _lc_to_chunks(lc_docs)
        assert isinstance(chunks[0].id, UUID)

    def test_invalid_document_id_fallback(self):
        lc_docs = [
            LC_Document(
                page_content="text",
                metadata={"document_id": "bad-uuid", "format": "txt"},
            ),
        ]
        chunks = _lc_to_chunks(lc_docs)
        assert chunks[0].document_id is None

    def test_invalid_format_raises(self):
        lc_docs = [
            LC_Document(
                page_content="text",
                metadata={"format": "nonexistent_format"},
            ),
        ]
        with pytest.raises(ValueError):
            _lc_to_chunks(lc_docs)

    def test_invalid_language_ignored(self):
        lc_docs = [
            LC_Document(
                page_content="text",
                metadata={"format": "txt", "language": "xyz"},
            ),
        ]
        with pytest.raises(ValueError):
            _lc_to_chunks(lc_docs)

    def test_empty_content(self):
        lc_docs = [
            LC_Document(
                page_content="",
                metadata={"format": "txt"},
            ),
        ]
        chunks = _lc_to_chunks(lc_docs)
        assert len(chunks) == 1
        assert chunks[0].text == ""
        assert chunks[0].start_char is None
        assert chunks[0].end_char is None


# ================================================================
# LangChainChunker.chunk  method tests
# ================================================================


class _TestChunker(LangChainChunker):
    def _splitter(self, config=None):
        return MagicMock()

    def _default_config(self):
        return ChunkerConfig()


class TestLangChainChunker:
    def test_chunk_returns_text_chunks(self, mocker):
        mock_splitter = MagicMock()
        mock_splitter.split_documents.return_value = [
            LC_Document(page_content="chunk1", metadata={"format": "txt"}),
            LC_Document(page_content="chunk2", metadata={"format": "txt"}),
        ]

        chunker = _TestChunker()
        mocker.patch.object(chunker, "_splitter", return_value=mock_splitter)
        docs = [TextDocument(text="full text")]
        result = chunker.chunk(docs, config=None)

        assert len(result) == 2
        assert all(isinstance(c, TextChunk) for c in result)

    def test_chunk_passes_config_to_splitter(self, mocker):
        mock_splitter = MagicMock()
        mock_splitter.split_documents.return_value = [
            LC_Document(page_content="c", metadata={"format": "txt"}),
        ]

        config = ChunkerConfig(chunk_size=256, chunk_overlap=32)
        chunker = _TestChunker()
        spy = mocker.patch.object(chunker, "_splitter", return_value=mock_splitter)
        chunker.chunk([TextDocument(text="some text")], config=config)

        spy.assert_called_once_with(config)

    def test_chunk_empty_documents(self, mocker):
        mock_splitter = MagicMock()
        mock_splitter.split_documents.return_value = []

        chunker = _TestChunker()
        mocker.patch.object(chunker, "_splitter", return_value=mock_splitter)
        result = chunker.chunk([], config=None)

        assert result == []

    def test_chunk_with_config_none(self, mocker):
        mock_splitter = MagicMock()
        mock_splitter.split_documents.return_value = [
            LC_Document(page_content="data", metadata={"format": "txt"}),
        ]

        chunker = _TestChunker()
        mocker.patch.object(chunker, "_splitter", return_value=mock_splitter)
        result = chunker.chunk([TextDocument(text="data")], config=None)

        assert len(result) == 1

    def test_chunk_multiple_documents(self, mocker):
        mock_splitter = MagicMock()
        mock_splitter.split_documents.return_value = [
            LC_Document(page_content="doc1-chunk", metadata={"format": "txt"}),
            LC_Document(page_content="doc2-chunk", metadata={"format": "txt"}),
        ]

        chunker = _TestChunker()
        mocker.patch.object(chunker, "_splitter", return_value=mock_splitter)
        docs = [TextDocument(text="doc1"), TextDocument(text="doc2")]
        result = chunker.chunk(docs, config=None)

        assert len(result) == 2

    def test_empty_text_document(self, mocker):
        mock_splitter = MagicMock()
        mock_splitter.split_documents.return_value = [
            LC_Document(page_content="", metadata={"format": "txt"}),
        ]

        chunker = _TestChunker()
        mocker.patch.object(chunker, "_splitter", return_value=mock_splitter)
        result = chunker.chunk([TextDocument(text="")], config=ChunkerConfig())

        assert len(result) == 1
        assert result[0].text == ""

    def test_splitter_error_propagates(self, mocker):
        chunker = _TestChunker()
        mock_splitter = MagicMock()
        mock_splitter.split_documents.side_effect = RuntimeError("split failed")

        mocker.patch.object(chunker, "_splitter", return_value=mock_splitter)
        with pytest.raises(RuntimeError, match="split failed"):
            chunker.chunk([TextDocument(text="fail")], config=ChunkerConfig())


# ================================================================
# Real-text integration test  (was completely missing)
# ================================================================


class TestRecursiveChunkerIntegration:
    @pytest.fixture
    def chunker(self):
        from agent_platform.integrations.chunking.recursive.recursive import (
            RecursiveChunkerProvider,
        )

        return RecursiveChunkerProvider()

    def test_splits_simple_text(self, chunker, simple_text_document):
        from agent_platform.integrations.chunking.recursive.config import (
            RecursiveChunkerConfig,
        )

        config = RecursiveChunkerConfig(chunk_size=20, chunk_overlap=0)
        result = chunker.chunk([simple_text_document], config=config)

        assert len(result) > 1
        assert all(isinstance(c, TextChunk) for c in result)
        assert all(len(c.text) <= 20 for c in result)
        assert result[0].document_id == simple_text_document.id

    def test_splits_long_text(self, chunker, long_text_document):
        from agent_platform.integrations.chunking.recursive.config import (
            RecursiveChunkerConfig,
        )

        config = RecursiveChunkerConfig(chunk_size=100, chunk_overlap=20)
        result = chunker.chunk([long_text_document], config=config)

        assert len(result) > 1
        assert all(len(c.text) <= 100 for c in result)
        for c in result:
            assert c.index >= 0
            assert isinstance(c.id, UUID)

    def test_uses_add_start_index(self, chunker, simple_text_document):
        from agent_platform.integrations.chunking.recursive.config import (
            RecursiveChunkerConfig,
        )

        config = RecursiveChunkerConfig(
            chunk_size=20, chunk_overlap=0, add_start_index=True
        )
        result = chunker.chunk([simple_text_document], config=config)

        assert any(c.start_char is not None for c in result)

    def test_respects_separators(self, chunker):
        doc = TextDocument(
            text="paragraph one with enough text to fill a chunk\n\nparagraph two also with enough text to fill another chunk\n\nparagraph three",
            source="test.txt",
            format=DocumentFormat.TXT,
        )
        from agent_platform.integrations.chunking.recursive.config import (
            RecursiveChunkerConfig,
        )

        config = RecursiveChunkerConfig(
            chunk_size=50, chunk_overlap=0, separators=["\n\n"]
        )
        result = chunker.chunk([doc], config=config)

        assert len(result) >= 2

    def test_empty_text(self, chunker):
        doc = TextDocument(text="", format=DocumentFormat.TXT)
        from agent_platform.integrations.chunking.recursive.config import (
            RecursiveChunkerConfig,
        )

        config = RecursiveChunkerConfig(chunk_size=100, chunk_overlap=0)
        result = chunker.chunk([doc], config=config)
        assert result == []
