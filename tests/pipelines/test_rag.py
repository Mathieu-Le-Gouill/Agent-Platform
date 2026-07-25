from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from agent_platform.core.interfaces.embeddings.response import EmbeddingResponse
from agent_platform.core.interfaces.llm.response import LLMResponse
from agent_platform.core.interfaces.vector_store.config import VectorStoreConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.document import TextDocument
from agent_platform.core.schemas.embedding import Embedding
from agent_platform.core.schemas.score import Score
from agent_platform.core.schemas.token import TokenUsage
from agent_platform.pipelines.rag.ingest import ingest
from agent_platform.pipelines.rag.query import query


class TestIngest:
    @pytest.fixture
    def sample_documents(self):
        return [TextDocument(id=uuid4(), text="doc 1")]

    @pytest.fixture
    def sample_chunks(self):
        return [
            TextChunk(id=uuid4(), text="chunk 1", index=0),
            TextChunk(id=uuid4(), text="chunk 2", index=1),
        ]

    @pytest.fixture
    def mock_loader(self, sample_documents):
        loader = AsyncMock()
        loader.arun = AsyncMock(return_value=sample_documents)
        return loader

    @pytest.fixture
    def mock_chunker(self, sample_chunks):
        chunker = AsyncMock()
        chunker.arun = AsyncMock(return_value=sample_chunks)
        return chunker

    @pytest.fixture
    def mock_store(self):
        store = AsyncMock()
        store.add = AsyncMock()
        return store

    async def test_ingest_calls_loader_chunker_store_in_order(
        self, mock_loader, mock_chunker, mock_store, sample_documents, sample_chunks
    ):
        sources = ["a.txt", "b.txt"]
        await ingest(sources, mock_loader, mock_chunker, mock_store)

        mock_loader.arun.assert_awaited_once_with(sources)
        mock_chunker.arun.assert_awaited_once_with(sample_documents)
        mock_store.add.assert_awaited_once_with(sample_chunks, config=None)

    async def test_ingest_passes_config_through(
        self, mock_loader, mock_chunker, mock_store, sample_chunks
    ):
        config = VectorStoreConfig(collection_name="docs")
        await ingest(["a.txt"], mock_loader, mock_chunker, mock_store, config=config)
        mock_store.add.assert_awaited_once_with(sample_chunks, config=config)

    async def test_ingest_empty_sources(self, mock_loader, mock_chunker, mock_store):
        mock_loader.arun = AsyncMock(return_value=[])
        mock_chunker.arun = AsyncMock(return_value=[])
        await ingest([], mock_loader, mock_chunker, mock_store)
        mock_store.add.assert_awaited_once_with([], config=None)


class TestQuery:
    @pytest.fixture
    def sample_chunks(self):
        return [
            TextChunk(id=uuid4(), text="result 1", index=0),
            TextChunk(id=uuid4(), text="result 2", index=1),
        ]

    @pytest.fixture
    def mock_embedder(self):
        embedder = AsyncMock()
        embedder.arun = AsyncMock(
            return_value=EmbeddingResponse(
                embeddings=[Embedding.from_list([0.1, 0.2, 0.3], model="test-model")],
                model="test-model",
            )
        )
        return embedder

    @pytest.fixture
    def mock_vector_search(self, sample_chunks):
        vector_search = AsyncMock()
        vector_search.arun = AsyncMock(
            return_value=[(chunk, Score(value=1.0)) for chunk in sample_chunks]
        )
        return vector_search

    @pytest.fixture
    def mock_reranker(self, sample_chunks):
        reranker = AsyncMock()
        reranker.arun = AsyncMock(return_value=sample_chunks)
        return reranker

    @pytest.fixture
    def mock_generator(self):
        generator = AsyncMock()
        generator.arun = AsyncMock(
            return_value=LLMResponse(
                message=None,
                usage=TokenUsage(),
                model="test-model",
            )
        )
        return generator

    async def test_query_calls_pipeline_in_order(
        self,
        mock_embedder,
        mock_vector_search,
        mock_reranker,
        mock_generator,
        sample_chunks,
    ):
        result = await query(
            "what is this?",
            mock_embedder,
            mock_vector_search,
            mock_reranker,
            mock_generator,
            k=5,
        )

        mock_embedder.arun.assert_awaited_once()
        mock_vector_search.arun.assert_awaited_once_with(([0.1, 0.2, 0.3], 5, None))
        mock_reranker.arun.assert_awaited_once_with(("what is this?", sample_chunks))
        mock_generator.arun.assert_awaited_once()
        assert result.model == "test-model"

    async def test_query_passes_filter_and_k_through(
        self, mock_embedder, mock_vector_search, mock_reranker, mock_generator
    ):
        await query(
            "q",
            mock_embedder,
            mock_vector_search,
            mock_reranker,
            mock_generator,
            k=3,
            filter={"source": "doc.txt"},
        )
        mock_vector_search.arun.assert_awaited_once_with(
            ([0.1, 0.2, 0.3], 3, {"source": "doc.txt"})
        )

    async def test_query_builds_prompt_from_reranked_context(
        self, mock_embedder, mock_vector_search, mock_reranker, mock_generator
    ):
        await query(
            "what is this?",
            mock_embedder,
            mock_vector_search,
            mock_reranker,
            mock_generator,
        )
        prompt = mock_generator.arun.await_args.args[0]
        system_text = prompt.system_prompt()
        assert "result 1" in system_text
        assert "result 2" in system_text
        assert prompt.last_user_message().text == "what is this?"
