from agent_platform.core.interfaces.embeddings.base import BaseEmbeddingProvider
from agent_platform.core.interfaces.embeddings.config import EmbeddingConfig
from agent_platform.core.interfaces.embeddings.response import EmbeddingResponse
from agent_platform.core.schemas.embedding import Embedding

# ================================================================
# EmbeddingResponse model tests
# ================================================================


class TestEmbeddingResponse:
    def test_iter_returns_embeddings(self):
        emb1 = Embedding(vector=(0.1, 0.2))
        emb2 = Embedding(vector=(0.3, 0.4))
        response = EmbeddingResponse(embeddings=[emb1, emb2], model="test")
        assert list(response) == [emb1, emb2]

    def test_len_returns_count(self):
        response = EmbeddingResponse(
            embeddings=[Embedding(vector=(0.1,))], model="test"
        )
        assert len(response) == 1

    def test_len_empty(self):
        response = EmbeddingResponse(embeddings=[], model="test")
        assert len(response) == 0

    def test_iter_over_empty(self):
        response = EmbeddingResponse(embeddings=[], model="test")
        assert list(response) == []

    def test_iter_multiple(self):
        embs = [Embedding(vector=(float(i),)) for i in range(5)]
        response = EmbeddingResponse(embeddings=embs, model="t")
        assert len(list(response)) == 5


class TestBaseEmbeddingProvider:
    def test_stores_credentials(self):
        marker = object()

        class _Concrete(BaseEmbeddingProvider[EmbeddingConfig]):
            def __init__(self, credentials):
                self._credentials = credentials

            def embed_document(self, items, config=None):
                return EmbeddingResponse(embeddings=[], model="")

            async def aembed_document(self, items, config=None):
                return EmbeddingResponse(embeddings=[], model="")

            def embed_query(self, query, config=None):
                return EmbeddingResponse(embeddings=[], model="")

            async def aembed_query(self, query, config=None):
                return EmbeddingResponse(embeddings=[], model="")

        provider = _Concrete(marker)
        assert provider._credentials is marker
