from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import SecretStr

pytest.importorskip("langchain_openai")
pytest.importorskip("langchain_mistralai")
pytest.importorskip("langchain_ollama")
pytest.importorskip("langchain_huggingface")

from agent_platform.integrations.embeddings.providers.openai import (
    OpenAIEmbeddingProvider,
    _to_langchain_openai,
)
from agent_platform.integrations.embeddings.providers.huggingface import (
    HuggingFaceEmbeddingProvider,
    _to_langchain_huggingface_local,
    _to_langchain_huggingface_hosted,
)
from agent_platform.integrations.embeddings.providers.mistral import (
    MistralEmbeddingProvider,
    _to_langchain_mistral,
)
from agent_platform.integrations.embeddings.providers.ollama import (
    OllamaEmbeddingProvider,
    _to_langchain_ollama,
)
from agent_platform.integrations.embeddings.openai.config import OpenAIEmbeddingConfig
from agent_platform.integrations.embeddings.mistral.config import MistralEmbeddingConfig
from agent_platform.integrations.embeddings.ollama.config import OllamaEmbeddingConfig
from agent_platform.integrations.embeddings.huggingface.config import (
    HuggingFaceEmbeddingConfig,
    HuggingFaceEmbeddingMode,
)
from agent_platform.core.interfaces.embeddings.response import EmbeddingResponse
from agent_platform.core.interfaces.embeddings.base import BaseEmbeddingProvider
from agent_platform.integrations.credentials import OpenAICredentials
from agent_platform.integrations.credentials import MistralCredentials
from agent_platform.integrations.credentials import OllamaCredentials
from agent_platform.integrations.credentials import HuggingFaceCredentials
from agent_platform.core.errors import MissingCredentialError
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.embedding import Embedding


# ================================================================
# _to_langchain_*  config mapping tests  (were completely missing)
# ================================================================


class TestToLangchainOpenAI:
    def test_default_model_kwargs(self):
        cfg = OpenAIEmbeddingConfig()
        creds = OpenAICredentials()
        result = _to_langchain_openai(cfg, creds)
        assert "model_kwargs" in result

    def test_dimensions_included_when_set(self):
        cfg = OpenAIEmbeddingConfig(dimensions=256)
        creds = OpenAICredentials()
        result = _to_langchain_openai(cfg, creds)
        assert result["dimensions"] == 256

    def test_dimensions_omitted_when_none(self):
        cfg = OpenAIEmbeddingConfig()
        creds = OpenAICredentials()
        result = _to_langchain_openai(cfg, creds)
        assert "dimensions" not in result

    def test_timeout_from_config(self):
        cfg = OpenAIEmbeddingConfig(timeout=15.0)
        creds = OpenAICredentials()
        result = _to_langchain_openai(cfg, creds)
        assert result["timeout"] == 15.0

    def test_timeout_falls_back_to_credentials(self):
        cfg = OpenAIEmbeddingConfig()
        creds = OpenAICredentials(timeout=30.0)
        result = _to_langchain_openai(cfg, creds)
        assert result["timeout"] == 30.0

    def test_max_retries_from_config(self):
        cfg = OpenAIEmbeddingConfig(max_retries=5)
        creds = OpenAICredentials()
        result = _to_langchain_openai(cfg, creds)
        assert result["max_retries"] == 5

    def test_max_retries_falls_back_to_credentials(self):
        cfg = OpenAIEmbeddingConfig()
        creds = OpenAICredentials()
        result = _to_langchain_openai(cfg, creds)
        assert result["max_retries"] == 3

    def test_organization_included_when_set(self):
        cfg = OpenAIEmbeddingConfig()
        creds = OpenAICredentials(organization="my-org")
        result = _to_langchain_openai(cfg, creds)
        assert result["organization"] == "my-org"

    def test_organization_omitted_when_none(self):
        cfg = OpenAIEmbeddingConfig()
        creds = OpenAICredentials()
        result = _to_langchain_openai(cfg, creds)
        assert "organization" not in result


class TestToLangchainMistral:
    def test_default_endpoint(self):
        cfg = MistralEmbeddingConfig()
        creds = MistralCredentials()
        result = _to_langchain_mistral(cfg, creds)
        assert result["endpoint"] == "https://api.mistral.ai/v1/"

    def test_timeout_from_config(self):
        cfg = MistralEmbeddingConfig(timeout=30.0)
        creds = MistralCredentials()
        result = _to_langchain_mistral(cfg, creds)
        assert isinstance(result["timeout"], int)
        assert result["timeout"] == 30

    def test_timeout_falls_back_to_credentials(self):
        cfg = MistralEmbeddingConfig()
        creds = MistralCredentials(timeout=60.0)
        result = _to_langchain_mistral(cfg, creds)
        assert result["timeout"] == 60

    def test_max_retries_fallback(self):
        cfg = MistralEmbeddingConfig()
        creds = MistralCredentials()
        result = _to_langchain_mistral(cfg, creds)
        assert result["max_retries"] == 3

    def test_dimensions_included_when_set(self):
        cfg = MistralEmbeddingConfig(dimensions=128)
        creds = MistralCredentials()
        result = _to_langchain_mistral(cfg, creds)
        assert result["dimensions"] == 128

    def test_dimensions_omitted_when_none(self):
        cfg = MistralEmbeddingConfig()
        creds = MistralCredentials()
        result = _to_langchain_mistral(cfg, creds)
        assert "dimensions" not in result

    def test_wait_time_included_when_set(self):
        cfg = MistralEmbeddingConfig(wait_time=5)
        creds = MistralCredentials()
        result = _to_langchain_mistral(cfg, creds)
        assert result["wait_time"] == 5

    def test_max_concurrent_requests_included_when_set(self):
        cfg = MistralEmbeddingConfig(max_concurrent_requests=10)
        creds = MistralCredentials()
        result = _to_langchain_mistral(cfg, creds)
        assert result["max_concurrent_requests"] == 10


class TestToLangchainOllama:
    def test_dimensions_omitted_when_none(self):
        cfg = OllamaEmbeddingConfig()
        creds = OllamaCredentials()
        result = _to_langchain_ollama(cfg, creds)
        assert "dimensions" not in result

    def test_dimensions_included_when_set(self):
        cfg = OllamaEmbeddingConfig(dimensions=384)
        creds = OllamaCredentials()
        result = _to_langchain_ollama(cfg, creds)
        assert result["dimensions"] == 384

    def test_temperature_included_when_set(self):
        cfg = OllamaEmbeddingConfig(temperature=0.3)
        creds = OllamaCredentials()
        result = _to_langchain_ollama(cfg, creds)
        assert result["temperature"] == 0.3

    def test_top_p_included_when_set(self):
        cfg = OllamaEmbeddingConfig(top_p=0.9)
        creds = OllamaCredentials()
        result = _to_langchain_ollama(cfg, creds)
        assert result["top_p"] == 0.9

    def test_top_k_included_when_set(self):
        cfg = OllamaEmbeddingConfig(top_k=40)
        creds = OllamaCredentials()
        result = _to_langchain_ollama(cfg, creds)
        assert result["top_k"] == 40

    def test_timeout_from_config(self):
        cfg = OllamaEmbeddingConfig(timeout=25.0)
        creds = OllamaCredentials()
        result = _to_langchain_ollama(cfg, creds)
        assert result["timeout"] == 25.0


class TestToLangchainHuggingFaceLocal:
    def test_empty_when_no_kwargs(self):
        cfg = HuggingFaceEmbeddingConfig()
        result = _to_langchain_huggingface_local(cfg)
        assert result == {}

    def test_model_kwargs_included(self):
        cfg = HuggingFaceEmbeddingConfig(model_kwargs={"device": "cpu"})
        result = _to_langchain_huggingface_local(cfg)
        assert result["model_kwargs"] == {"device": "cpu"}

    def test_encode_kwargs_included(self):
        cfg = HuggingFaceEmbeddingConfig(encode_kwargs={"show_progress_bar": True})
        result = _to_langchain_huggingface_local(cfg)
        assert result["encode_kwargs"] == {"show_progress_bar": True}

    def test_both_kwargs_included(self):
        cfg = HuggingFaceEmbeddingConfig(
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 16},
        )
        result = _to_langchain_huggingface_local(cfg)
        assert result["model_kwargs"] == {"device": "cpu"}
        assert result["encode_kwargs"] == {"batch_size": 16}


class TestToLangchainHuggingFaceHosted:
    def test_empty_when_no_fields(self):
        cfg = HuggingFaceEmbeddingConfig()
        creds = HuggingFaceCredentials()
        result = _to_langchain_huggingface_hosted(cfg, creds)
        assert result == {}

    def test_provider_included_when_set(self):
        cfg = HuggingFaceEmbeddingConfig(provider="huggingface")
        creds = HuggingFaceCredentials()
        result = _to_langchain_huggingface_hosted(cfg, creds)
        assert result["provider"] == "huggingface"

    def test_timeout_from_config(self):
        cfg = HuggingFaceEmbeddingConfig(timeout=20.0)
        creds = HuggingFaceCredentials()
        result = _to_langchain_huggingface_hosted(cfg, creds)
        assert result["timeout"] == 20.0


# ================================================================
# MissingCredentialError tests
# ================================================================


class TestMissingCredentialError:
    def test_openai_missing_api_key(self):
        cfg = OpenAIEmbeddingConfig(model="text-embedding-ada-002")
        provider = OpenAIEmbeddingProvider()
        provider._credentials = OpenAICredentials()
        with pytest.raises(MissingCredentialError, match="OpenAI API key is required"):
            provider._client(cfg)

    def test_mistral_missing_api_key(self):
        cfg = MistralEmbeddingConfig(model="mistral-embed")
        provider = MistralEmbeddingProvider()
        provider._credentials = MistralCredentials()
        with pytest.raises(MissingCredentialError, match="Mistral API key is required"):
            provider._client(cfg)

    def test_huggingface_hosted_missing_api_key(self):
        cfg = HuggingFaceEmbeddingConfig(
            model="some-model",
            mode=HuggingFaceEmbeddingMode.HOSTED,
        )
        provider = HuggingFaceEmbeddingProvider()
        provider._credentials = HuggingFaceCredentials()
        with pytest.raises(
            MissingCredentialError, match="Hugging Face Hub API token is required"
        ):
            provider._client(cfg)

    def test_ollama_no_credential_needed(self):
        cfg = OllamaEmbeddingConfig(model="nomic-embed-text")
        provider = OllamaEmbeddingProvider()
        client = provider._client(cfg)
        assert client is not None


# ================================================================
# HuggingFace _client mode branch tests
# ================================================================


class TestHuggingFaceClient:
    @patch(
        "agent_platform.integrations.embeddings.providers.huggingface.HuggingFaceEmbeddings"
    )
    def test_local_mode_calls_local_embeddings(self, mock_local):
        cfg = HuggingFaceEmbeddingConfig(mode=HuggingFaceEmbeddingMode.LOCAL)
        provider = HuggingFaceEmbeddingProvider()
        provider._client(cfg)
        mock_local.assert_called_once()

    @patch(
        "agent_platform.integrations.embeddings.providers.huggingface.HuggingFaceEndpointEmbeddings"
    )
    def test_hosted_mode_calls_hosted_embeddings(self, mock_hosted):
        cfg = HuggingFaceEmbeddingConfig(
            mode=HuggingFaceEmbeddingMode.HOSTED,
            model="bert-base-uncased",
        )
        creds = HuggingFaceCredentials(api_key=SecretStr("hf_test"))
        provider = HuggingFaceEmbeddingProvider(credentials=creds)
        provider._client(cfg)
        mock_hosted.assert_called_once()


# ================================================================
# Provider-level embed method tests  (sync embed_document,
# sync/async embed_query were completely missing)
# ================================================================


class TestEmbedDocumentSync:
    def test_returns_embedding_response(self, openai_provider):
        provider, mock_lc = openai_provider
        uid1, uid2 = uuid4(), uuid4()
        items = [
            TextChunk(id=uid1, text="hello"),
            TextChunk(id=uid2, text="world"),
        ]
        mock_lc.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
        config = MagicMock(model="test-model")

        response = provider.embed_document(items, config=config)

        assert isinstance(response, EmbeddingResponse)
        assert len(response) == 2
        assert response.model == "test-model"
        assert response.embeddings[0].id == uid1
        assert response.embeddings[0].to_list() == [0.1, 0.2]
        mock_lc.embed_documents.assert_called_once_with(["hello", "world"])

    def test_empty_list(self, openai_provider):
        provider, mock_lc = openai_provider
        mock_lc.embed_documents.return_value = []

        response = provider.embed_document([], config=MagicMock(model="m"))

        assert len(response) == 0
        assert response.embeddings == []


class TestEmbedQuerySync:
    def test_returns_single_embedding(self, openai_provider):
        provider, mock_lc = openai_provider
        mock_lc.embed_query.return_value = [0.5, 0.6, 0.7]
        config = MagicMock(model="test-model")

        response = provider.embed_query("hello world", config=config)

        assert isinstance(response, EmbeddingResponse)
        assert len(response) == 1
        assert response.model == "test-model"
        assert response.embeddings[0].to_list() == [0.5, 0.6, 0.7]
        mock_lc.embed_query.assert_called_once_with("hello world")


class TestEmbedQueryAsync:
    async def test_returns_single_embedding(self, openai_provider):
        provider, mock_lc = openai_provider
        mock_lc.aembed_query.return_value = [0.8, 0.9]
        config = MagicMock(model="async-model")

        response = await provider.aembed_query("test query", config=config)

        assert isinstance(response, EmbeddingResponse)
        assert len(response) == 1
        assert response.model == "async-model"
        assert response.embeddings[0].to_list() == [0.8, 0.9]
        mock_lc.aembed_query.assert_awaited_once_with("test query")

    async def test_query_propagates_empty_vector(self, openai_provider):
        provider, mock_lc = openai_provider
        mock_lc.aembed_query.return_value = []
        config = MagicMock(model="m")

        with pytest.raises(ValueError, match="cannot be empty"):
            await provider.aembed_query("", config=config)


# ================================================================
# aembed_document tests (refactored: no more object.__new__)
# ================================================================


class TestAembedDocument:
    async def test_returns_embedding_response(self, openai_provider):
        provider, mock_lc = openai_provider
        uid1, uid2 = uuid4(), uuid4()
        items = [
            TextChunk(id=uid1, text="hello"),
            TextChunk(id=uid2, text="world"),
        ]
        mock_lc.aembed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
        config = MagicMock(model="test-model")

        response = await provider.aembed_document(items, config=config)

        assert isinstance(response, EmbeddingResponse)
        assert len(response) == 2
        assert response.model == "test-model"
        assert response.embeddings[0].id == uid1
        assert response.embeddings[0].to_list() == [0.1, 0.2]
        assert response.embeddings[1].id == uid2
        assert response.embeddings[1].to_list() == [0.3, 0.4]
        mock_lc.aembed_documents.assert_awaited_once_with(["hello", "world"])

    async def test_single_item(self, openai_provider):
        provider, mock_lc = openai_provider
        uid = uuid4()
        items = [TextChunk(id=uid, text="single")]
        mock_lc.aembed_documents.return_value = [[0.5, 0.6]]

        response = await provider.aembed_document(items, config=MagicMock(model="t"))

        assert len(response) == 1
        assert response.embeddings[0].to_list() == [0.5, 0.6]

    async def test_empty_list(self, openai_provider):
        provider, mock_lc = openai_provider
        mock_lc.aembed_documents.return_value = []

        response = await provider.aembed_document([], config=MagicMock(model="m"))

        assert len(response) == 0
        assert response.embeddings == []

    async def test_multiple_texts(self, openai_provider):
        provider, mock_lc = openai_provider
        items = [
            TextChunk(id=uuid4(), text="text A"),
            TextChunk(id=uuid4(), text="text B"),
            TextChunk(id=uuid4(), text="text C"),
        ]
        mock_lc.aembed_documents.return_value = [
            [0.0, 0.1],
            [0.2, 0.3],
            [0.4, 0.5],
        ]

        response = await provider.aembed_document(items, config=MagicMock(model="m"))

        assert len(response) == 3
        assert response.embeddings[0].to_list() == [0.0, 0.1]
        assert response.embeddings[2].to_list() == [0.4, 0.5]


# ================================================================
# Error handling tests  (were completely missing)
# ================================================================


class TestErrorHandling:
    async def test_langchain_raise_in_async_embed(self, openai_provider):
        provider, mock_lc = openai_provider
        mock_lc.aembed_documents.side_effect = RuntimeError("API error")
        items = [TextChunk(id=uuid4(), text="fail")]

        with pytest.raises(RuntimeError, match="API error"):
            await provider.aembed_document(items, config=MagicMock(model="m"))

    def test_langchain_raise_in_sync_embed(self, openai_provider):
        provider, mock_lc = openai_provider
        mock_lc.embed_documents.side_effect = ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            provider.embed_document(
                [TextChunk(id=uuid4(), text="fail")],
                config=MagicMock(model="m"),
            )


# ================================================================
# Constructor / _default_config tests  (consolidated)
# ================================================================


class TestProviderDefaults:
    def test_openai_default_config_model(self):
        provider = OpenAIEmbeddingProvider()
        assert provider._default_config().model == "text-embedding-ada-002"

    def test_huggingface_default_config_model(self):
        provider = HuggingFaceEmbeddingProvider()
        assert (
            provider._default_config().model == "sentence-transformers/all-MiniLM-L6-v2"
        )

    def test_mistral_default_config_model(self):
        provider = MistralEmbeddingProvider()
        assert provider._default_config().model == "mistral-embed"

    def test_ollama_default_config_model(self):
        provider = OllamaEmbeddingProvider()
        assert provider._default_config().model == "nomic-embed-text"


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

        class _Concrete(BaseEmbeddingProvider[OpenAIEmbeddingConfig]):
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
