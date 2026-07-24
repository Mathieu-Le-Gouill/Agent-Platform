import pytest

from agent_platform.components.vector_search import VectorSearch
from agent_platform.core.interfaces.vector_store.config import VectorStoreConfig
from agent_platform.core.interfaces.vector_store.port import VectorStore
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import Score


class _FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.calls: list[tuple[list[float], int, dict | None]] = []

    async def add(self, documents, config=None):
        raise NotImplementedError

    async def delete(self, document_ids, config=None):
        raise NotImplementedError

    async def search(self, query_vector, k=5, config=None, filter=None):
        raise NotImplementedError

    async def search_with_scores(self, query_vector, k=5, config=None, filter=None):
        self.calls.append((query_vector, k, config, filter))
        return [(TextChunk(text="match", index=0), Score.similarity(0.9))]


@pytest.mark.asyncio
async def test_forwards_vector_k_and_filter_to_backend():
    backend = _FakeVectorStore()
    config = VectorStoreConfig(collection_name="docs")
    vector_search = VectorSearch(backend, config)

    result = await vector_search.arun(([0.1, 0.2], 3, {"source": "wiki"}))

    assert result[0][0].text == "match"
    assert backend.calls[0] == ([0.1, 0.2], 3, config, {"source": "wiki"})
