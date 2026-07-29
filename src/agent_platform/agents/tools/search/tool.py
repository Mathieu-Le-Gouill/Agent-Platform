from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from agent_platform.agents.tools._utils import safe_call
from agent_platform.agents.tools.base import Tool, ToolError
from agent_platform.components.embedder.component import Embedder
from agent_platform.components.vector_search.component import VectorSearch
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import Score


class SearchInput(BaseModel):
    query: str = Field(..., min_length=1, description="Text query for semantic search")
    k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of results to return",
    )


class SearchResult(BaseModel):
    chunk: TextChunk
    score: Score


class SearchTool(Tool):
    name = "search"
    description = (
        "Search indexed documents by semantic similarity to a text query. "
        "Returns ranked text chunks with relevance scores."
    )
    input_schema = SearchInput
    output_schema = SearchResult
    supports_streaming = True

    def __init__(
        self,
        embedder: Embedder,
        vector_search: VectorSearch,
    ) -> None:
        self._embedder = embedder
        self._vector_search = vector_search

    async def _search(self, validated: SearchInput) -> list[SearchResult]:
        query_chunk = TextChunk(id=uuid4(), text=validated.query, index=0)
        response = await safe_call(
            self._embedder.arun([query_chunk]),
            "Embedding failed",
        )
        if not response.embeddings:
            raise ToolError("Embedding returned no vectors")
        vector = response.embeddings[0].to_list()
        results = await safe_call(
            self._vector_search.arun((vector, validated.k, None)),
            "Vector search failed",
        )
        return [SearchResult(chunk=chunk, score=score) for chunk, score in results]

    async def run(self, **kwargs: Any) -> list[SearchResult]:
        return await self._search(SearchInput(**kwargs))

    async def astream(self, **kwargs: Any) -> AsyncIterator[str]:
        for result in await self._search(SearchInput(**kwargs)):
            yield result.model_dump_json()
