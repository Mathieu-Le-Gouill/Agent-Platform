import pytest

from agent_platform.components.embedder import Embedder
from agent_platform.core.interfaces.embeddings.base import BaseEmbeddingProvider
from agent_platform.core.interfaces.embeddings.config import EmbeddingConfig
from agent_platform.core.interfaces.embeddings.response import EmbeddingResponse
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.embedding import Embedding


class _FakeEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self) -> None:
        self.batches: list[list[TextChunk]] = []

    def embed_document(self, items, config=None):
        raise NotImplementedError

    async def aembed_document(self, items, config=None):
        self.batches.append(list(items))
        return EmbeddingResponse(
            embeddings=[Embedding(vector=(float(len(c.text)),)) for c in items],
            model="fake-model",
        )

    def embed_query(self, query, config=None):
        raise NotImplementedError

    async def aembed_query(self, query, config=None):
        raise NotImplementedError


def _chunks(n: int) -> list[TextChunk]:
    return [TextChunk(text="x" * (i + 1), index=i) for i in range(n)]


@pytest.mark.asyncio
async def test_single_batch_when_under_limit():
    backend = _FakeEmbeddingProvider()
    embedder = Embedder(backend, EmbeddingConfig(batch_size=10))

    response = await embedder.arun(_chunks(3))

    assert len(backend.batches) == 1
    assert len(response.embeddings) == 3
    assert response.model == "fake-model"


@pytest.mark.asyncio
async def test_splits_into_multiple_batches_over_limit():
    backend = _FakeEmbeddingProvider()
    embedder = Embedder(backend, EmbeddingConfig(batch_size=2))

    chunks = _chunks(5)
    response = await embedder.arun(chunks)

    assert [len(b) for b in backend.batches] == [2, 2, 1]
    assert len(response.embeddings) == 5


@pytest.mark.asyncio
async def test_merged_output_preserves_input_order():
    backend = _FakeEmbeddingProvider()
    embedder = Embedder(backend, EmbeddingConfig(batch_size=2))

    chunks = _chunks(5)
    response = await embedder.arun(chunks)

    assert [e.vector[0] for e in response.embeddings] == [
        float(len(c.text)) for c in chunks
    ]


@pytest.mark.asyncio
async def test_defaults_to_config_batch_size_when_none_given():
    backend = _FakeEmbeddingProvider()
    embedder = Embedder(backend, config=None)

    await embedder.arun(_chunks(EmbeddingConfig().batch_size + 1))

    assert len(backend.batches) == 2
