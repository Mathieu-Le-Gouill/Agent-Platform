import pytest

from agent_platform.components.embedder.component import Embedder
from agent_platform.components.semantic_chunker.component import SemanticChunker
from agent_platform.components.semantic_chunker.config import SemanticChunkerConfig
from agent_platform.core.interfaces.embeddings.response import EmbeddingResponse
from agent_platform.core.schemas.document import TextDocument
from agent_platform.core.schemas.embedding import Embedding


class _FakeEmbedder(Embedder):
    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors

    async def arun(self, input):
        chunks, _config = input
        vectors = self._vectors[: len(chunks)]
        return EmbeddingResponse(
            embeddings=[Embedding.from_list(v) for v in vectors],
            model="fake",
        )


@pytest.mark.asyncio
async def test_single_sentence_returns_one_chunk():
    embedder = _FakeEmbedder([[1.0, 0.0]])
    chunker = SemanticChunker(embedder)
    doc = TextDocument(text="Only one sentence here.")

    chunks = await chunker.arun(([doc], None))

    assert len(chunks) == 1
    assert chunks[0].text == "Only one sentence here."
    assert chunks[0].document_id == doc.id


@pytest.mark.asyncio
async def test_empty_document_returns_no_chunks():
    embedder = _FakeEmbedder([])
    chunker = SemanticChunker(embedder)
    doc = TextDocument(text="   ")

    chunks = await chunker.arun(([doc], None))

    assert chunks == []


@pytest.mark.asyncio
async def test_breaks_on_low_similarity():
    # Two clusters of near-identical vectors, orthogonal to each other.
    vectors = [
        [1.0, 0.0],
        [0.99, 0.01],
        [0.0, 1.0],
        [0.01, 0.99],
    ]
    embedder = _FakeEmbedder(vectors)
    config = SemanticChunkerConfig(breakpoint_percentile_threshold=50.0)
    chunker = SemanticChunker(embedder)
    doc = TextDocument(
        text="First sentence. Second sentence. Third sentence. Fourth sentence."
    )

    chunks = await chunker.arun(([doc], config))

    assert len(chunks) >= 2
    assert all(c.metadata["chunking_strategy"] == "semantic" for c in chunks)


@pytest.mark.asyncio
async def test_multiple_documents_are_chunked_independently():
    embedder = _FakeEmbedder([[1.0, 0.0], [0.9, 0.1], [1.0, 0.0], [0.9, 0.1]])
    chunker = SemanticChunker(embedder)
    doc1 = TextDocument(text="One. Two.")
    doc2 = TextDocument(text="Three. Four.")

    chunks = await chunker.arun(([doc1, doc2], None))

    doc_ids = {c.document_id for c in chunks}
    assert doc1.id in doc_ids
    assert doc2.id in doc_ids
