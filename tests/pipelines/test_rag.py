from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from agent_platform.models.chunk import TextChunk
from agent_platform.pipelines.rag.ingest import ingest
from agent_platform.pipelines.rag.query import query


class TestIngest:
    @pytest.fixture
    def mock_store(self):
        store = AsyncMock()
        store.add = AsyncMock()
        return store

    @pytest.fixture
    def sample_chunks(self):
        return [
            TextChunk(id=uuid4(), text="chunk 1", index=0),
            TextChunk(id=uuid4(), text="chunk 2", index=1),
        ]

    async def test_ingest_calls_store_add(self, mock_store, sample_chunks):
        await ingest(sample_chunks, mock_store)
        mock_store.add.assert_awaited_once_with(sample_chunks)

    async def test_ingest_empty_list(self, mock_store):
        await ingest([], mock_store)
        mock_store.add.assert_awaited_once_with([])

    async def test_ingest_single_chunk(self, mock_store):
        chunk = TextChunk(id=uuid4(), text="single", index=0)
        await ingest([chunk], mock_store)
        mock_store.add.assert_awaited_once_with([chunk])


class TestQuery:
    @pytest.fixture
    def mock_store(self):
        store = AsyncMock()
        store.search = AsyncMock(
            return_value=[
                TextChunk(id=uuid4(), text="result 1", index=0),
                TextChunk(id=uuid4(), text="result 2", index=1),
            ]
        )
        return store

    async def test_query_returns_chunks(self, mock_store):
        results = await query([0.1, 0.2, 0.3], mock_store, k=5)
        assert len(results) == 2
        assert results[0].text == "result 1"
        assert results[1].text == "result 2"

    async def test_query_default_k(self, mock_store):
        await query([0.1, 0.2, 0.3], mock_store)
        mock_store.search.assert_awaited_once_with([0.1, 0.2, 0.3], k=5)

    async def test_query_custom_k(self, mock_store):
        await query([0.1, 0.2, 0.3], mock_store, k=10)
        mock_store.search.assert_awaited_once_with([0.1, 0.2, 0.3], k=10)

    async def test_query_empty_vector(self, mock_store):
        await query([], mock_store)
        mock_store.search.assert_awaited_once_with([], k=5)

    async def test_query_empty_result(self):
        store = AsyncMock()
        store.search = AsyncMock(return_value=[])
        results = await query([0.1, 0.2], store)
        assert results == []

    async def test_query_passes_vector_correctly(self, mock_store):
        vector = [0.5, 0.6, 0.7, 0.8]
        await query(vector, mock_store, k=3)
        mock_store.search.assert_awaited_once_with(vector, k=3)
