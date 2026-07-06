from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from agent_platform.integrations.vector_store.langchain_base import (
    _chunk_to_lc,
    _lc_to_chunk,
    LangChainVectorStore,
)
from agent_platform.models.chunk import TextChunk
from agent_platform.models.enums import DocumentFormat, Language
from agent_platform.models.score import Score


def test_chunk_to_lc_basic():
    uid = uuid4()
    doc_id = uuid4()
    chunk = TextChunk(
        id=uid,
        document_id=doc_id,
        text="Hello world",
        index=0,
        start_char=0,
        end_char=11,
        format=DocumentFormat.TXT,
        metadata={"source": "test.txt", "language": "en", "extra": {"key": "val"}},
    )
    lc = _chunk_to_lc(chunk)
    assert lc.page_content == "Hello world"
    assert lc.metadata["document_id"] == str(doc_id)
    assert lc.metadata["index"] == 0
    assert lc.metadata["chunk_id"] == str(uid)
    assert lc.metadata["start_char"] == 0
    assert lc.metadata["end_char"] == 11
    assert lc.metadata["format"] == "txt"
    assert lc.metadata["source"] == "test.txt"
    assert lc.metadata["language"] == "en"
    assert lc.metadata["extra"] == {"key": "val"}


def test_chunk_to_lc_no_metadata():
    chunk = TextChunk(id=uuid4(), text="No metadata", index=1)
    lc = _chunk_to_lc(chunk)
    assert lc.metadata["document_id"] is None
    assert lc.metadata["source"] is None
    assert lc.metadata["language"] is None
    assert lc.metadata["extra"] is None


def test_chunk_to_lc_default_format():
    chunk = TextChunk(id=uuid4(), text="test", index=0)
    lc = _chunk_to_lc(chunk)
    assert lc.metadata["format"] == "unknown"


def test_lc_to_chunk_basic():
    uid = uuid4()
    doc_id = uuid4()
    lc_doc = type(
        "LCDoc",
        (),
        {
            "page_content": "Hello world",
            "metadata": {
                "chunk_id": str(uid),
                "document_id": str(doc_id),
                "index": 0,
                "start_char": 0,
                "end_char": 11,
                "format": "txt",
                "source": "test.txt",
                "language": "en",
                "extra": {"key": "val"},
            },
        },
    )()
    chunk = _lc_to_chunk(lc_doc)
    assert chunk.id == uid
    assert chunk.document_id == doc_id
    assert chunk.text == "Hello world"
    assert chunk.index == 0
    assert chunk.metadata["source"] == "test.txt"
    assert chunk.metadata["language"] == Language.EN
    assert chunk.metadata["extra"] == {"key": "val"}


def test_lc_to_chunk_no_chunk_id_generates_new():
    lc_doc = type(
        "LCDoc",
        (),
        {
            "page_content": "orphan",
            "metadata": {"chunk_id": None, "document_id": None, "index": 0},
        },
    )()
    chunk = _lc_to_chunk(lc_doc)
    assert chunk.id is not None
    assert chunk.document_id is None


def test_lc_to_chunk_null_metadata():
    lc_doc = type(
        "LCDoc",
        (),
        {
            "page_content": "data",
            "metadata": {
                "chunk_id": None,
                "document_id": None,
                "index": 0,
                "start_char": None,
                "end_char": None,
                "format": None,
                "source": None,
                "language": None,
                "extra": None,
            },
        },
    )()
    chunk = _lc_to_chunk(lc_doc)
    assert chunk.text == "data"
    assert chunk.metadata["source"] is None
    assert chunk.metadata["language"] is None
    assert chunk.metadata["extra"] == {}


def test_lc_to_chunk_missing_index_defaults_zero():
    lc_doc = type(
        "LCDoc",
        (),
        {
            "page_content": "data",
            "metadata": {
                "chunk_id": None,
                "document_id": None,
                "start_char": None,
                "end_char": None,
                "format": None,
                "source": None,
                "language": None,
                "extra": None,
            },
        },
    )()
    chunk = _lc_to_chunk(lc_doc)
    assert chunk.index == 0


def test_lc_to_chunk_language_enum():
    lc_doc = type(
        "LCDoc",
        (),
        {
            "page_content": "Bonjour",
            "metadata": {
                "chunk_id": None,
                "document_id": None,
                "index": 0,
                "start_char": None,
                "end_char": None,
                "format": None,
                "source": None,
                "language": "fr",
                "extra": None,
            },
        },
    )()
    chunk = _lc_to_chunk(lc_doc)
    assert chunk.metadata["language"] == Language.FR


def test_round_trip():
    uid = uuid4()
    doc_id = uuid4()
    original = TextChunk(
        id=uid,
        document_id=doc_id,
        text="Round trip test",
        index=2,
        start_char=0,
        end_char=15,
        format=DocumentFormat.MARKDOWN,
        metadata={"source": "doc.md", "language": "fr", "extra": {"line": 42}},
    )
    lc = _chunk_to_lc(original)
    restored = _lc_to_chunk(lc)
    assert restored.id == original.id
    assert restored.document_id == original.document_id
    assert restored.text == original.text
    assert restored.index == original.index
    assert restored.metadata["source"] == original.metadata["source"]
    assert restored.metadata["language"] == Language.FR
    assert restored.metadata["extra"] == original.metadata["extra"]


class _TestVectorStore(LangChainVectorStore):
    def _build_client(self):
        return MagicMock()

    async def delete(self, document_ids):  # type: ignore[override]
        pass


class TestLangChainVectorStore:
    async def test_add_calls_aadd_documents(self):
        uid = uuid4()
        doc = TextChunk(id=uid, text="hello", index=0)

        mock_client = MagicMock()
        mock_client.aadd_documents = MagicMock()

        store = _TestVectorStore()
        store._client = mock_client

        await store.add([doc])

        mock_client.aadd_documents.assert_called_once()
        args = mock_client.aadd_documents.call_args[0][0]
        assert len(args) == 1
        assert args[0].page_content == "hello"

    async def test_add_multiple_documents(self):
        docs = [
            TextChunk(id=uuid4(), text="first", index=0),
            TextChunk(id=uuid4(), text="second", index=1),
        ]

        mock_client = MagicMock()
        mock_client.aadd_documents = MagicMock()

        store = _TestVectorStore()
        store._client = mock_client

        await store.add(docs)

        args = mock_client.aadd_documents.call_args[0][0]
        assert len(args) == 2

    async def test_search_returns_text_chunks(self):
        uid = uuid4()
        mock_lc_doc = type(
            "LCDoc",
            (),
            {
                "page_content": "found doc",
                "metadata": {
                    "chunk_id": str(uid),
                    "document_id": None,
                    "index": 0,
                    "start_index": None,
                    "format": None,
                    "source": None,
                    "language": None,
                    "extra": None,
                },
            },
        )()

        mock_client = MagicMock()
        mock_client.similarity_search_by_vector = MagicMock(return_value=[mock_lc_doc])

        store = _TestVectorStore()
        store._client = mock_client

        results = await store.search(query_vector=[0.1, 0.2, 0.3], k=5)

        assert len(results) == 1
        assert isinstance(results[0], TextChunk)
        assert results[0].id == uid
        assert results[0].text == "found doc"

    async def test_search_default_k(self):
        mock_client = MagicMock()
        mock_client.similarity_search_by_vector = MagicMock(return_value=[])

        store = _TestVectorStore()
        store._client = mock_client

        await store.search(query_vector=[0.1, 0.2])

        mock_client.similarity_search_by_vector.assert_called_once()
        _, kwargs = mock_client.similarity_search_by_vector.call_args
        assert (
            kwargs.get("k") == 5
            or mock_client.similarity_search_by_vector.call_args[0][1] == 5
        )

    async def test_search_with_scores_returns_tuples(self):
        uid = uuid4()
        mock_lc_doc = type(
            "LCDoc",
            (),
            {
                "page_content": "scored doc",
                "metadata": {
                    "chunk_id": str(uid),
                    "document_id": None,
                    "index": 0,
                    "start_index": None,
                    "format": None,
                    "source": None,
                    "language": None,
                    "extra": None,
                },
            },
        )()

        mock_client = MagicMock()
        mock_client.similarity_search_with_score = MagicMock(
            return_value=[(mock_lc_doc, 0.85)],
        )

        store = _TestVectorStore()
        store._client = mock_client

        results = await store.search_with_scores(query_vector=[0.1, 0.2, 0.3], k=5)

        assert len(results) == 1
        chunk, score = results[0]
        assert isinstance(chunk, TextChunk)
        assert chunk.text == "scored doc"
        assert isinstance(score, Score)
        assert score.value == 0.85

    async def test_search_with_scores_multiple_results(self):
        uids = [uuid4(), uuid4()]
        mock_docs = [
            type(
                "LCDoc",
                (),
                {
                    "page_content": "a",
                    "metadata": {
                        "chunk_id": str(uids[0]),
                        "document_id": None,
                        "index": 0,
                        "start_index": None,
                        "format": None,
                        "source": None,
                        "language": None,
                        "extra": None,
                    },
                },
            )(),
            type(
                "LCDoc",
                (),
                {
                    "page_content": "b",
                    "metadata": {
                        "chunk_id": str(uids[1]),
                        "document_id": None,
                        "index": 1,
                        "start_index": None,
                        "format": None,
                        "source": None,
                        "language": None,
                        "extra": None,
                    },
                },
            )(),
        ]

        mock_client = MagicMock()
        mock_client.similarity_search_with_score = MagicMock(
            return_value=[(mock_docs[0], 0.9), (mock_docs[1], 0.7)],
        )

        store = _TestVectorStore()
        store._client = mock_client

        results = await store.search_with_scores(query_vector=[0.1, 0.2])

        assert len(results) == 2
        assert results[0][1].value == 0.9
        assert results[1][1].value == 0.7

    async def test_search_with_scores_default_k(self):
        mock_client = MagicMock()
        mock_client.similarity_search_with_score = MagicMock(return_value=[])

        store = _TestVectorStore()
        store._client = mock_client

        await store.search_with_scores(query_vector=[0.1, 0.2])

        mock_client.similarity_search_with_score.assert_called_once()
        args = mock_client.similarity_search_with_score.call_args
        assert args[0][1] == 5 or args[1].get("k") == 5
