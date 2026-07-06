from __future__ import annotations

from pydantic import BaseModel, Field

from agent_platform.agents.tools.base import Tool, ToolError
from agent_platform.agents.tools._utils import safe_call, text_chunk
from agent_platform.integrations.embeddings.base import BaseEmbeddingProvider
from agent_platform.integrations.vector_store.port import VectorStore
from agent_platform.models.chunk import TextChunk
from agent_platform.models.score import Score


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

    def __init__(
        self,
        embedder: BaseEmbeddingProvider,
        store: VectorStore,
    ) -> None:
        self._embedder = embedder
        self._store = store

    async def run(self, **kwargs) -> list[SearchResult]:
        validated = SearchInput(**kwargs)
        query_chunk = text_chunk(validated.query)
        response = await safe_call(
            self._embedder.encode([query_chunk]),
            "Embedding failed",
        )
        if not response.embeddings:
            raise ToolError("Embedding returned no vectors")
        vector = response.embeddings[0].to_list()
        results = await safe_call(
            self._store.search_with_scores(vector, k=validated.k),
            "Vector search failed",
        )
        return [SearchResult(chunk=chunk, score=score) for chunk, score in results]
