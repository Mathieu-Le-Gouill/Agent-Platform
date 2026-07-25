from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from langchain_core.documents import Document as LC_Document

from agent_platform.core.errors import ProviderError
from agent_platform.core.interfaces.reranking.config import RerankerConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.enums import DocumentFormat, Language
from agent_platform.integrations.reranking.langchain_base import (
    LangChainReranker,
    _chunk_to_lc,
    _lc_to_chunks,
)


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


def test_lc_to_chunks_single():
    uid = uuid4()
    doc_id = uuid4()
    lc = LC_Document(
        page_content="Hello world",
        metadata={
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
    )
    chunks = _lc_to_chunks([lc])
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.id == uid
    assert chunk.document_id == doc_id
    assert chunk.text == "Hello world"
    assert chunk.index == 0
    assert chunk.metadata["source"] == "test.txt"
    assert chunk.metadata["language"] == Language.EN
    assert chunk.metadata["extra"] == {"key": "val"}


def test_lc_to_chunks_multiple():
    uid1, uid2 = uuid4(), uuid4()
    lc1 = LC_Document(
        page_content="First",
        metadata={
            "chunk_id": str(uid1),
            "document_id": None,
            "index": 0,
            "format": None,
            "source": None,
            "language": None,
            "extra": None,
        },
    )
    lc2 = LC_Document(
        page_content="Second",
        metadata={
            "chunk_id": str(uid2),
            "document_id": None,
            "index": 1,
            "format": None,
            "source": None,
            "language": None,
            "extra": None,
        },
    )
    chunks = _lc_to_chunks([lc1, lc2])
    assert len(chunks) == 2
    assert chunks[0].id == uid1
    assert chunks[1].id == uid2
    assert chunks[0].text == "First"
    assert chunks[1].text == "Second"


def test_lc_to_chunks_no_chunk_id():
    lc = LC_Document(
        page_content="orphan",
        metadata={
            "chunk_id": None,
            "document_id": None,
            "index": 0,
        },
    )
    chunks = _lc_to_chunks([lc])
    assert chunks[0].id is not None


def test_lc_to_chunks_null_metadata():
    lc = LC_Document(
        page_content="data",
        metadata={
            "chunk_id": None,
            "document_id": None,
            "index": 0,
            "format": None,
            "source": None,
            "language": None,
            "extra": None,
        },
    )
    chunks = _lc_to_chunks([lc])
    assert chunks[0].metadata["source"] is None
    assert chunks[0].metadata["language"] is None
    assert chunks[0].metadata["extra"] == {}


def test_lc_to_chunks_handles_language_enum():
    lc = LC_Document(
        page_content="Hola",
        metadata={
            "chunk_id": None,
            "document_id": None,
            "index": 0,
            "format": None,
            "source": None,
            "language": "es",
            "extra": None,
        },
    )
    chunks = _lc_to_chunks([lc])
    assert chunks[0].metadata["language"] == Language.SP


def test_lc_to_chunks_no_config_no_score():
    lc = LC_Document(
        page_content="a",
        metadata={
            "chunk_id": str(uuid4()),
            "document_id": None,
            "index": 0,
            "relevance_score": 0.9,
        },
    )
    chunks = _lc_to_chunks([lc])
    assert chunks[0].confidence is None


def test_lc_to_chunks_return_scores_false_no_score():
    lc = LC_Document(
        page_content="a",
        metadata={
            "chunk_id": str(uuid4()),
            "document_id": None,
            "index": 0,
            "relevance_score": 0.9,
        },
    )
    cfg = RerankerConfig(return_scores=False)
    chunks = _lc_to_chunks([lc], cfg)
    assert chunks[0].confidence is None


def test_lc_to_chunks_return_scores_normalized():
    lc1 = LC_Document(
        page_content="a",
        metadata={
            "chunk_id": str(uuid4()),
            "document_id": None,
            "index": 0,
            "relevance_score": 0.2,
        },
    )
    lc2 = LC_Document(
        page_content="b",
        metadata={
            "chunk_id": str(uuid4()),
            "document_id": None,
            "index": 1,
            "relevance_score": 0.8,
        },
    )
    cfg = RerankerConfig(return_scores=True, normalize_scores=True)
    chunks = _lc_to_chunks([lc1, lc2], cfg)
    assert chunks[0].confidence.value == 0.0
    assert chunks[1].confidence.value == 1.0
    assert chunks[0].confidence.low == 0.0
    assert chunks[0].confidence.high == 1.0


def test_lc_to_chunks_return_scores_raw_unbounded():
    lc = LC_Document(
        page_content="a",
        metadata={
            "chunk_id": str(uuid4()),
            "document_id": None,
            "index": 0,
            "relevance_score": 5.3,
        },
    )
    cfg = RerankerConfig(return_scores=True, normalize_scores=False)
    chunks = _lc_to_chunks([lc], cfg)
    assert chunks[0].confidence.value == 5.3
    assert chunks[0].confidence.low == float("-inf")
    assert chunks[0].confidence.high == float("inf")


def test_lc_to_chunks_return_scores_missing_metadata_key():
    lc = LC_Document(
        page_content="a",
        metadata={"chunk_id": str(uuid4()), "document_id": None, "index": 0},
    )
    cfg = RerankerConfig(return_scores=True)
    chunks = _lc_to_chunks([lc], cfg)
    assert chunks[0].confidence is None


def test_round_trip():
    uid = uuid4()
    original = TextChunk(
        id=uid,
        text="Round trip test",
        index=2,
        start_char=0,
        end_char=15,
        format=DocumentFormat.MARKDOWN,
        metadata={"source": "doc.md", "language": "fr", "extra": {"line": 42}},
    )
    lc = _chunk_to_lc(original)
    [restored] = _lc_to_chunks([lc])
    assert restored.id == original.id
    assert restored.text == original.text
    assert restored.index == original.index
    assert restored.metadata["source"] == original.metadata["source"]
    assert restored.metadata["language"] == Language.FR
    assert restored.metadata["extra"] == original.metadata["extra"]


class _TestReranker(LangChainReranker):
    def _default_config(self):
        from agent_platform.core.interfaces.reranking.config import RerankerConfig

        return RerankerConfig()

    def _client(self, config):
        return MagicMock()


class TestLangChainReranker:
    async def test_rerank_returns_text_chunks(self, mocker):
        uid = uuid4()
        item = TextChunk(id=uid, text="test doc", index=0)

        result_uid = uuid4()
        mock_result_lc = LC_Document(
            page_content="reranked result",
            metadata={
                "chunk_id": str(result_uid),
                "document_id": str(uid),
                "index": 0,
                "start_index": None,
                "format": None,
                "source": None,
                "language": None,
                "extra": None,
            },
        )

        mock_client = MagicMock()
        mock_client.acompress_documents = AsyncMock(return_value=[mock_result_lc])

        reranker = _TestReranker()
        mocker.patch.object(reranker, "_client", return_value=mock_client)
        results = await reranker.rerank(query="test query", items=[item])

        assert len(results) == 1
        assert isinstance(results[0], TextChunk)
        assert results[0].id == result_uid
        assert results[0].text == "reranked result"

    async def test_rerank_with_top_k(self, mocker):
        items = [TextChunk(id=uuid4(), text=f"doc{i}", index=i) for i in range(5)]
        mock_results_lc = [
            LC_Document(
                page_content=f"result{i}",
                metadata={
                    "chunk_id": str(uuid4()),
                    "document_id": None,
                    "index": i,
                    "start_index": None,
                    "format": None,
                    "source": None,
                    "language": None,
                    "extra": None,
                },
            )
            for i in range(5)
        ]

        mock_client = MagicMock()
        mock_client.acompress_documents = AsyncMock(return_value=mock_results_lc)

        config = RerankerConfig(top_k=3)

        reranker = _TestReranker()
        mocker.patch.object(reranker, "_client", return_value=mock_client)
        results = await reranker.rerank(query="q", items=items, config=config)

        assert len(results) == 3

    async def test_rerank_empty_items(self, mocker):
        mock_client = MagicMock()
        mock_client.acompress_documents = AsyncMock(return_value=[])

        reranker = _TestReranker()
        mocker.patch.object(reranker, "_client", return_value=mock_client)
        results = await reranker.rerank(query="q", items=[])

        assert results == []

    async def test_rerank_calls_acompress_documents(self, mocker):
        uid = uuid4()
        item = TextChunk(id=uid, text="hello", index=0)

        mock_client = MagicMock()
        mock_client.acompress_documents = AsyncMock(return_value=[])

        reranker = _TestReranker()
        mocker.patch.object(reranker, "_client", return_value=mock_client)
        await reranker.rerank(query="the query", items=[item])

        mock_client.acompress_documents.assert_awaited_once()
        args, _ = mock_client.acompress_documents.await_args
        assert len(args[0]) == 1
        assert args[0][0].page_content == "hello"
        assert args[1] == "the query"

    async def test_rerank_without_top_k_returns_all(self, mocker):
        items = [TextChunk(id=uuid4(), text=f"doc{i}", index=i) for i in range(3)]
        mock_results_lc = [
            LC_Document(
                page_content=f"result{i}",
                metadata={
                    "chunk_id": str(uuid4()),
                    "document_id": None,
                    "index": i,
                    "start_index": None,
                    "format": None,
                    "source": None,
                    "language": None,
                    "extra": None,
                },
            )
            for i in range(3)
        ]

        mock_client = MagicMock()
        mock_client.acompress_documents = AsyncMock(return_value=mock_results_lc)

        config = RerankerConfig(top_k=None)

        reranker = _TestReranker()
        mocker.patch.object(reranker, "_client", return_value=mock_client)
        results = await reranker.rerank(query="q", items=items, config=config)

        assert len(results) == 3

    async def test_rerank_propagates_relevance_score(self, mocker):
        items = [TextChunk(id=uuid4(), text="a", index=0)]
        mock_result_lc = LC_Document(
            page_content="a",
            metadata={
                "chunk_id": str(uuid4()),
                "document_id": None,
                "index": 0,
                "start_index": None,
                "format": None,
                "source": None,
                "language": None,
                "extra": None,
                "relevance_score": 0.42,
            },
        )
        mock_client = MagicMock()
        mock_client.acompress_documents = AsyncMock(return_value=[mock_result_lc])

        config = RerankerConfig(return_scores=True, normalize_scores=False)
        reranker = _TestReranker()
        mocker.patch.object(reranker, "_client", return_value=mock_client)
        results = await reranker.rerank(query="q", items=items, config=config)

        assert results[0].confidence is not None
        assert results[0].confidence.value == 0.42

    async def test_rerank_return_scores_false_leaves_confidence_none(self, mocker):
        items = [TextChunk(id=uuid4(), text="a", index=0)]
        mock_result_lc = LC_Document(
            page_content="a",
            metadata={
                "chunk_id": str(uuid4()),
                "document_id": None,
                "index": 0,
                "start_index": None,
                "format": None,
                "source": None,
                "language": None,
                "extra": None,
                "relevance_score": 0.42,
            },
        )
        mock_client = MagicMock()
        mock_client.acompress_documents = AsyncMock(return_value=[mock_result_lc])

        reranker = _TestReranker()
        mocker.patch.object(reranker, "_client", return_value=mock_client)
        results = await reranker.rerank(query="q", items=items)

        assert results[0].confidence is None


class TestRerankRetryAndTranslation:
    async def test_retries_transient_failure_then_succeeds(
        self, no_retry_sleep, mocker
    ):
        calls = {"n": 0}

        async def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("transient")
            return []

        mock_client = MagicMock()
        mock_client.acompress_documents = flaky

        reranker = _TestReranker()
        mocker.patch.object(reranker, "_client", return_value=mock_client)
        await reranker.rerank(query="q", items=[])

        assert calls["n"] == 2

    async def test_translates_permanent_failure_to_provider_error(
        self, no_retry_sleep, mocker
    ):
        async def always_fails(*args, **kwargs):
            raise ConnectionError("boom")

        mock_client = MagicMock()
        mock_client.acompress_documents = always_fails

        reranker = _TestReranker()
        mocker.patch.object(reranker, "_client", return_value=mock_client)
        with pytest.raises(ProviderError, match="Reranking failed"):
            await reranker.rerank(query="q", items=[])
