from uuid import uuid4

from agent_platform.core.schemas.document import DocumentMetadata, TextDocument
from agent_platform.core.schemas.enums import DocumentFormat, Language
from agent_platform.pipelines.ingestion import chunk_document


def _make_doc(
    text: str,
    source: str = "",
    title: str | None = None,
    language: Language | None = None,
):
    return TextDocument(
        id=uuid4(),
        text=text,
        source=source,
        format=DocumentFormat.TXT,
        metadata=DocumentMetadata(title=title),
        language=language,
    )


class TestChunkDocument:
    def test_empty_document_returns_empty_list(self):
        doc = _make_doc("")
        assert chunk_document(doc) == []

    def test_document_with_no_text_attribute_fallthrough(self):
        doc = _make_doc("")
        result = chunk_document(doc)
        assert result == []

    def test_short_text_creates_single_chunk(self):
        doc = _make_doc("Hello world")
        chunks = chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world"
        assert chunks[0].index == 0

    def test_long_text_creates_multiple_chunks(self):
        text = " ".join(["word"] * 2000)
        doc = _make_doc(text)
        chunks = chunk_document(doc)
        assert len(chunks) > 1
        assert all(c.index == i for i, c in enumerate(chunks))
        assert all(c.document_id == doc.id for c in chunks)

    def test_metadata_propagates_source(self):
        doc = _make_doc("Hello world", source="test.txt")
        chunks = chunk_document(doc)
        assert chunks[0].metadata["source"] == "test.txt"

    def test_metadata_propagates_language(self):
        doc = _make_doc("Bonjour le monde", language=Language.FR)
        chunks = chunk_document(doc)
        assert chunks[0].metadata["language"] == "fr"

    def test_metadata_language_none_when_no_language(self):
        doc = _make_doc("Hello", language=None)
        chunks = chunk_document(doc)
        assert chunks[0].metadata["language"] is None

    def test_metadata_propagates_title(self):
        doc = _make_doc("Hello", title="My Document")
        chunks = chunk_document(doc)
        assert chunks[0].metadata["title"] == "My Document"

    def test_metadata_title_none(self):
        doc = _make_doc("Hello", title=None)
        chunks = chunk_document(doc)
        assert chunks[0].metadata["title"] is None

    def test_chunks_have_unique_ids(self):
        text = " ".join(["word"] * 2000)
        doc = _make_doc(text)
        chunks = chunk_document(doc)
        ids = [c.id for c in chunks]
        assert len(set(ids)) == len(ids)

    def test_all_chunks_reference_correct_document_id(self):
        text = " ".join(["word"] * 2000)
        doc = _make_doc(text)
        chunks = chunk_document(doc)
        for c in chunks:
            assert c.document_id == doc.id
