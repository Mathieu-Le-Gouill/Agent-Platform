from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from agent_platform.integrations.embeddings.providers.openai import (
    OpenAIEmbeddingProvider,
)
from agent_platform.integrations.embeddings.providers.sentence_transformer import (
    SentenceTransformerEmbeddingProvider,
)
from agent_platform.integrations.embeddings.providers.mistral import (
    MistralEmbeddingProvider,
)
from agent_platform.integrations.embeddings.providers.ollama import (
    OllamaEmbeddingProvider,
)
from agent_platform.integrations.embeddings.response import EmbeddingResponse
from agent_platform.integrations.embeddings.langchain_base import LangChainEmbedder
from agent_platform.models.chunk import TextChunk
from agent_platform.models.token import TokenUsage
from agent_platform.models.embedding import Embedding


class TestOpenAIEmbeddingProvider:
    @patch.object(OpenAIEmbeddingProvider, "_build_client")
    def test_construct_with_defaults(self, mock_build):
        mock_build.return_value = MagicMock()
        provider = OpenAIEmbeddingProvider()
        assert provider._model == "text-embedding-3-small"
        assert provider.config is not None

    @patch.object(OpenAIEmbeddingProvider, "_build_client")
    def test_construct_with_custom(self, mock_build):
        mock_build.return_value = MagicMock()
        provider = OpenAIEmbeddingProvider(model="text-embedding-3-large")
        assert provider._model == "text-embedding-3-large"


class TestSentenceTransformerEmbeddingProvider:
    @patch.object(SentenceTransformerEmbeddingProvider, "_build_client")
    def test_construct_with_defaults(self, mock_build):
        mock_build.return_value = MagicMock()
        provider = SentenceTransformerEmbeddingProvider()
        assert provider._model == "sentence-transformers/all-MiniLM-L6-v2"

    @patch.object(SentenceTransformerEmbeddingProvider, "_build_client")
    def test_construct_with_custom(self, mock_build):
        mock_build.return_value = MagicMock()
        provider = SentenceTransformerEmbeddingProvider(
            model="sentence-transformers/all-mpnet-base-v2",
        )
        assert provider._model == "sentence-transformers/all-mpnet-base-v2"


class TestMistralEmbeddingProvider:
    @patch.object(MistralEmbeddingProvider, "_build_client")
    def test_construct_with_defaults(self, mock_build):
        mock_build.return_value = MagicMock()
        provider = MistralEmbeddingProvider()
        assert provider._model == "mistral-embed"

    @patch.object(MistralEmbeddingProvider, "_build_client")
    def test_construct_with_custom(self, mock_build):
        mock_build.return_value = MagicMock()
        provider = MistralEmbeddingProvider(model="mistral-embed-2")
        assert provider._model == "mistral-embed-2"


class TestOllamaEmbeddingProvider:
    @patch.object(OllamaEmbeddingProvider, "_build_client")
    def test_construct_with_defaults(self, mock_build):
        mock_build.return_value = MagicMock()
        provider = OllamaEmbeddingProvider()
        assert provider._model == "nomic-embed-text"

    @patch.object(OllamaEmbeddingProvider, "_build_client")
    def test_construct_with_custom(self, mock_build):
        mock_build.return_value = MagicMock()
        provider = OllamaEmbeddingProvider(model="llama-embed")
        assert provider._model == "llama-embed"


class TestLangChainEmbedderEncode:
    async def test_encode_returns_embedding_response(self):
        uid1 = uuid4()
        uid2 = uuid4()
        items = [
            TextChunk(id=uid1, text="hello"),
            TextChunk(id=uid2, text="world"),
        ]

        mock_client = MagicMock()
        mock_client.aembed_documents = AsyncMock(
            return_value=[[0.1, 0.2], [0.3, 0.4]],
        )

        provider = object.__new__(OpenAIEmbeddingProvider)
        provider._client = mock_client
        provider._model = "test-model"
        provider.config = None

        response = await provider.encode(items)

        assert isinstance(response, EmbeddingResponse)
        assert len(response) == 2
        assert response.model == "test-model"
        assert response.embeddings[0].id == uid1
        assert response.embeddings[0].to_list() == [0.1, 0.2]
        assert response.embeddings[1].id == uid2
        assert response.embeddings[1].to_list() == [0.3, 0.4]

        mock_client.aembed_documents.assert_awaited_once_with(["hello", "world"])

    async def test_encode_single_item(self):
        uid = uuid4()
        items = [TextChunk(id=uid, text="single")]

        mock_client = MagicMock()
        mock_client.aembed_documents = AsyncMock(return_value=[[0.5, 0.6]])

        provider = object.__new__(OllamaEmbeddingProvider)
        provider._client = mock_client
        provider._model = "test"

        response = await provider.encode(items)
        assert len(response) == 1
        assert response.embeddings[0].to_list() == [0.5, 0.6]

    async def test_encode_empty_list(self):
        mock_client = MagicMock()
        mock_client.aembed_documents = AsyncMock(return_value=[])

        provider = object.__new__(OpenAIEmbeddingProvider)
        provider._client = mock_client
        provider._model = "test"

        response = await provider.encode([])
        assert len(response) == 0
        assert response.embeddings == []

    async def test_encode_extract_usage_default(self):
        uid = uuid4()
        items = [TextChunk(id=uid, text="hello")]

        mock_client = MagicMock()
        mock_client.aembed_documents = AsyncMock(return_value=[[0.1, 0.2]])

        provider = object.__new__(OpenAIEmbeddingProvider)
        provider._client = mock_client
        provider._model = "test"
        provider.config = None

        response = await provider.encode(items)

        assert response.usage == TokenUsage.zero()

    async def test_encode_with_multiple_texts(self):
        items = [
            TextChunk(id=uuid4(), text="text A"),
            TextChunk(id=uuid4(), text="text B"),
            TextChunk(id=uuid4(), text="text C"),
        ]

        mock_client = MagicMock()
        mock_client.aembed_documents = AsyncMock(
            return_value=[[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]],
        )

        provider = object.__new__(OpenAIEmbeddingProvider)
        provider._client = mock_client
        provider._model = "m"
        provider.config = None

        response = await provider.encode(items)

        assert len(response) == 3
        assert response.embeddings[0].to_list() == [0.0, 0.1]
        assert response.embeddings[2].to_list() == [0.4, 0.5]

    async def test_extract_usage_is_called(self):
        uid = uuid4()
        items = [TextChunk(id=uid, text="x")]

        mock_client = MagicMock()
        mock_client.aembed_documents = AsyncMock(return_value=[[0.5]])

        with patch.object(
            LangChainEmbedder,
            "_extract_usage",
            return_value=TokenUsage(input_tokens=1, output_tokens=2),
        ) as mock_extract:
            provider = object.__new__(OpenAIEmbeddingProvider)
            provider._client = mock_client
            provider._model = "m"
            provider.config = None

            response = await provider.encode(items)

            mock_extract.assert_called_once_with([[0.5]])
            assert response.usage.input_tokens == 1
            assert response.usage.output_tokens == 2


class TestEmbeddingResponse:
    def test_iter_returns_embeddings(self):
        emb1 = Embedding(vector=(0.1, 0.2))
        emb2 = Embedding(vector=(0.3, 0.4))
        response = EmbeddingResponse(
            embeddings=[emb1, emb2],
            model="test",
            usage=TokenUsage.zero(),
        )
        result = list(response)
        assert result == [emb1, emb2]

    def test_len_returns_count(self):
        response = EmbeddingResponse(
            embeddings=[Embedding(vector=(0.1,))],
            model="test",
            usage=TokenUsage.zero(),
        )
        assert len(response) == 1

    def test_len_empty(self):
        response = EmbeddingResponse(
            embeddings=[],
            model="test",
            usage=TokenUsage.zero(),
        )
        assert len(response) == 0

    def test_iter_over_empty(self):
        response = EmbeddingResponse(
            embeddings=[],
            model="test",
            usage=TokenUsage.zero(),
        )
        assert list(response) == []

    def test_iter_multiple(self):
        embs = [Embedding(vector=(float(i),)) for i in range(5)]
        response = EmbeddingResponse(
            embeddings=embs,
            model="t",
            usage=TokenUsage.zero(),
        )
        assert len(list(response)) == 5
