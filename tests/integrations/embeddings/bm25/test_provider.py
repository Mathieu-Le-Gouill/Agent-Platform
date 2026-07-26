from agent_platform.integrations.embeddings.bm25.config import BM25Config
from agent_platform.integrations.embeddings.bm25.provider import BM25SparseEmbedder


class TestBM25SparseEmbedder:
    async def test_embed_sparse_returns_term_frequencies(self):
        embedder = BM25SparseEmbedder()
        result = await embedder.embed_sparse("the cat sat on the mat")

        assert len(result.indices) == len(result.values)
        assert len(result.indices) == 5  # unique tokens: the, cat, sat, on, mat
        assert sum(result.values) == 6  # total tokens including repeated "the"

    async def test_repeated_token_has_count_two(self):
        embedder = BM25SparseEmbedder()
        result = await embedder.embed_sparse("the the")

        assert len(result.indices) == 1
        assert result.values[0] == 2.0

    async def test_empty_text_returns_empty_vector(self):
        embedder = BM25SparseEmbedder()
        result = await embedder.embed_sparse("")

        assert result.indices == []
        assert result.values == []

    async def test_deterministic_across_calls(self):
        embedder = BM25SparseEmbedder()
        first = await embedder.embed_sparse("hello world")
        second = await embedder.embed_sparse("hello world")

        assert first.indices == second.indices
        assert first.values == second.values

    async def test_case_insensitive(self):
        embedder = BM25SparseEmbedder()
        lower = await embedder.embed_sparse("hello")
        upper = await embedder.embed_sparse("HELLO")

        assert lower.indices == upper.indices
        assert lower.values == upper.values

    async def test_respects_vocabulary_size_config(self):
        embedder = BM25SparseEmbedder()
        config = BM25Config(vocabulary_size=4)
        result = await embedder.embed_sparse("alpha beta gamma delta", config=config)

        assert all(0 <= index < 4 for index in result.indices)

    async def test_indices_sorted(self):
        embedder = BM25SparseEmbedder()
        result = await embedder.embed_sparse("zebra apple mango banana")

        assert result.indices == sorted(result.indices)
