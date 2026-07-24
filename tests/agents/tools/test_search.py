from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from agent_platform.agents.tools import (
    SearchInput,
    SearchResult,
    SearchTool,
    ToolError,
)
from agent_platform.core.interfaces.embeddings.response import EmbeddingResponse
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.embedding import Embedding
from agent_platform.core.schemas.score import Score


@pytest.fixture
def mock_embedder():
    embedder = AsyncMock()
    embedder.aembed_document = AsyncMock(
        return_value=EmbeddingResponse(
            embeddings=[Embedding.from_list([0.1, 0.2, 0.3])],
            model="test-model",
        )
    )
    return embedder


@pytest.fixture
def mock_store():
    store = AsyncMock()
    store.search_with_scores = AsyncMock(
        return_value=[
            (
                TextChunk(id=uuid4(), text="result 1", index=0),
                Score.similarity(0.95),
            ),
            (
                TextChunk(id=uuid4(), text="result 2", index=1),
                Score.similarity(0.85),
            ),
        ]
    )
    return store


@pytest.fixture
def tool(mock_embedder, mock_store):
    return SearchTool(embedder=mock_embedder, store=mock_store)


class TestSearchInput:
    def test_valid_input(self):
        inp = SearchInput(query="hello world")
        assert inp.query == "hello world"
        assert inp.k == 5

    def test_empty_query_raises(self):
        with pytest.raises(ValidationError):
            SearchInput(query="")

    def test_k_too_low(self):
        with pytest.raises(ValidationError):
            SearchInput(query="test", k=0)

    def test_k_too_high(self):
        with pytest.raises(ValidationError):
            SearchInput(query="test", k=101)

    def test_custom_k(self):
        inp = SearchInput(query="test", k=10)
        assert inp.k == 10


class TestSearchResult:
    def test_construct(self):
        chunk = TextChunk(id=uuid4(), text="test", index=0)
        score = Score.similarity(0.9)
        result = SearchResult(chunk=chunk, score=score)
        assert result.chunk.text == "test"
        assert result.score.value == 0.9


class TestSearchTool:
    def test_name_and_description(self, tool):
        assert tool.name == "search"
        assert tool.description

    def test_input_schema(self, tool):
        assert tool.input_schema is SearchInput

    def test_output_schema(self, tool):
        assert tool.output_schema is SearchResult

    @pytest.mark.asyncio
    async def test_run_success(self, tool, mock_embedder, mock_store):
        results = await tool.run(query="test query")
        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].chunk.text == "result 1"
        assert results[0].score.value == 0.95
        mock_embedder.aembed_document.assert_awaited_once()
        mock_store.search_with_scores.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_with_custom_k(self, tool, mock_embedder, mock_store):
        await tool.run(query="test", k=3)
        call_vector, call_kwargs = mock_store.search_with_scores.call_args
        assert call_kwargs["k"] == 3

    @pytest.mark.asyncio
    async def test_run_empty_embeddings_raises(self, tool, mock_embedder, mock_store):
        mock_embedder.aembed_document = AsyncMock(
            return_value=EmbeddingResponse(embeddings=[], model="test")
        )
        with pytest.raises(ToolError, match="no vectors"):
            await tool.run(query="test")

    @pytest.mark.asyncio
    async def test_run_empty_results(self, tool, mock_embedder, mock_store):
        mock_store.search_with_scores = AsyncMock(return_value=[])
        results = await tool.run(query="test")
        assert results == []

    @pytest.mark.asyncio
    async def test_run_embedder_error_wrapped(self, tool, mock_embedder, mock_store):
        mock_embedder.aembed_document = AsyncMock(
            side_effect=RuntimeError("embed failed")
        )
        with pytest.raises(ToolError, match="Embedding failed"):
            await tool.run(query="test")

    @pytest.mark.asyncio
    async def test_run_store_error_wrapped(self, tool, mock_embedder, mock_store):
        mock_store.search_with_scores = AsyncMock(
            side_effect=RuntimeError("store failed")
        )
        with pytest.raises(ToolError, match="Vector search failed"):
            await tool.run(query="test")

    @pytest.mark.asyncio
    async def test_run_missing_query_raises(self, tool):
        with pytest.raises(ValidationError):
            await tool.run(k=5)
