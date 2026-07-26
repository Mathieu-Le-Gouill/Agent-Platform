from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from pydantic import SecretStr

from agent_platform.core.errors import MissingCredentialError, ProviderError
from agent_platform.core.interfaces.embeddings.response import EmbeddingResponse
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.integrations.credentials import MistralCredentials
from agent_platform.integrations.embeddings.mistral.config import MistralEmbeddingConfig
from agent_platform.integrations.embeddings.mistral.provider import (
    MistralEmbeddingProvider,
)


def _creds(key: str = "mk-test") -> MistralCredentials:
    return MistralCredentials(api_key=SecretStr(key))


def _embeddings_response(vectors: list[list[float]]) -> SimpleNamespace:
    return SimpleNamespace(
        data=[SimpleNamespace(embedding=v, index=i) for i, v in enumerate(vectors)]
    )


class TestMistralEmbeddingProviderConstruction:
    def test_default_config_model(self):
        provider = MistralEmbeddingProvider()
        assert provider._default_config().model == "mistral-embed"

    def test_missing_api_key(self):
        provider = MistralEmbeddingProvider()
        provider._credentials = MistralCredentials()
        cfg = MistralEmbeddingConfig(model="mistral-embed")
        with pytest.raises(MissingCredentialError, match="Mistral API key is required"):
            provider._client(cfg)


class TestMistralEmbeddingProviderClient:
    def test_client_creation(self, mocker):
        mock_mistral = mocker.patch(
            "agent_platform.integrations.embeddings.mistral.provider.Mistral"
        )
        provider = MistralEmbeddingProvider(_creds())
        cfg = MistralEmbeddingConfig(timeout=30.0)
        client = provider._client(cfg)
        assert client is mock_mistral.return_value
        _, kwargs = mock_mistral.call_args
        assert kwargs["server_url"] == "https://api.mistral.ai/v1/"
        assert kwargs["timeout_ms"] == 30000

    def test_client_custom_endpoint(self, mocker):
        mock_mistral = mocker.patch(
            "agent_platform.integrations.embeddings.mistral.provider.Mistral"
        )
        provider = MistralEmbeddingProvider(_creds())
        provider._client(MistralEmbeddingConfig(endpoint="http://localhost:8080"))
        _, kwargs = mock_mistral.call_args
        assert kwargs["server_url"] == "http://localhost:8080"

    def test_params_dimensions_forwarded(self):
        provider = MistralEmbeddingProvider(_creds())
        params = provider._params(MistralEmbeddingConfig(dimensions=128))
        assert params["output_dimension"] == 128

    def test_params_dimensions_omitted_when_none(self):
        provider = MistralEmbeddingProvider(_creds())
        params = provider._params(MistralEmbeddingConfig())
        assert "output_dimension" not in params

    def test_vector_raises_on_missing_embedding(self):
        provider = MistralEmbeddingProvider(_creds())
        with pytest.raises(ProviderError, match="no embedding vector"):
            provider._vector(SimpleNamespace(embedding=None))

    def test_params_extra_params_applied(self):
        provider = MistralEmbeddingProvider(_creds())
        params = provider._params(
            MistralEmbeddingConfig(extra_params={"encoding_format": "base64"})
        )
        assert params["encoding_format"] == "base64"


class TestMistralEmbedDocument:
    def test_sync(self, mocker):
        mock_mistral = mocker.patch(
            "agent_platform.integrations.embeddings.mistral.provider.Mistral"
        )
        mock_client = MagicMock()
        mock_mistral.return_value = mock_client
        mock_client.embeddings.create.return_value = _embeddings_response(
            [[0.1, 0.2], [0.3, 0.4]]
        )

        uid1, uid2 = uuid4(), uuid4()
        items = [TextChunk(id=uid1, text="hello"), TextChunk(id=uid2, text="world")]
        provider = MistralEmbeddingProvider(_creds())
        response = provider.embed_document(items, config=MistralEmbeddingConfig())

        assert isinstance(response, EmbeddingResponse)
        assert len(response) == 2
        assert response.embeddings[0].id == uid1
        assert response.embeddings[0].to_list() == [0.1, 0.2]
        _, kwargs = mock_client.embeddings.create.call_args
        assert kwargs["inputs"] == ["hello", "world"]

    def test_empty_list(self, mocker):
        mock_mistral = mocker.patch(
            "agent_platform.integrations.embeddings.mistral.provider.Mistral"
        )
        mock_client = MagicMock()
        mock_mistral.return_value = mock_client
        mock_client.embeddings.create.return_value = _embeddings_response([])

        provider = MistralEmbeddingProvider(_creds())
        response = provider.embed_document([], config=MistralEmbeddingConfig())
        assert len(response) == 0

    async def test_async(self, mocker):
        mock_mistral = mocker.patch(
            "agent_platform.integrations.embeddings.mistral.provider.Mistral"
        )
        mock_client = MagicMock()
        mock_mistral.return_value = mock_client
        mock_client.embeddings.create_async = AsyncMock(
            return_value=_embeddings_response([[0.5, 0.6]])
        )

        uid = uuid4()
        items = [TextChunk(id=uid, text="single")]
        provider = MistralEmbeddingProvider(_creds())
        response = await provider.aembed_document(
            items, config=MistralEmbeddingConfig()
        )

        assert len(response) == 1
        assert response.embeddings[0].to_list() == [0.5, 0.6]

    async def test_async_translates_permanent_failure(self, mocker, no_retry_sleep):
        mock_mistral = mocker.patch(
            "agent_platform.integrations.embeddings.mistral.provider.Mistral"
        )
        mock_client = MagicMock()
        mock_mistral.return_value = mock_client

        async def always_fails(*args, **kwargs):
            raise ConnectionError("boom")

        mock_client.embeddings.create_async = always_fails

        provider = MistralEmbeddingProvider(_creds())
        with pytest.raises(ProviderError, match="Embedding generation failed"):
            await provider.aembed_document(
                [TextChunk(id=uuid4(), text="fail")],
                config=MistralEmbeddingConfig(),
            )


class TestMistralEmbedQuery:
    def test_sync(self, mocker):
        mock_mistral = mocker.patch(
            "agent_platform.integrations.embeddings.mistral.provider.Mistral"
        )
        mock_client = MagicMock()
        mock_mistral.return_value = mock_client
        mock_client.embeddings.create.return_value = _embeddings_response(
            [[0.5, 0.6, 0.7]]
        )

        provider = MistralEmbeddingProvider(_creds())
        response = provider.embed_query("hello world", config=MistralEmbeddingConfig())

        assert len(response) == 1
        assert response.embeddings[0].to_list() == [0.5, 0.6, 0.7]
        _, kwargs = mock_client.embeddings.create.call_args
        assert kwargs["inputs"] == ["hello world"]

    async def test_async(self, mocker):
        mock_mistral = mocker.patch(
            "agent_platform.integrations.embeddings.mistral.provider.Mistral"
        )
        mock_client = MagicMock()
        mock_mistral.return_value = mock_client
        mock_client.embeddings.create_async = AsyncMock(
            return_value=_embeddings_response([[0.8, 0.9]])
        )

        provider = MistralEmbeddingProvider(_creds())
        response = await provider.aembed_query(
            "test query", config=MistralEmbeddingConfig()
        )

        assert len(response) == 1
        assert response.embeddings[0].to_list() == [0.8, 0.9]

    async def test_async_retries_transient_failure_then_succeeds(
        self, mocker, no_retry_sleep
    ):
        mock_mistral = mocker.patch(
            "agent_platform.integrations.embeddings.mistral.provider.Mistral"
        )
        mock_client = MagicMock()
        mock_mistral.return_value = mock_client
        calls = {"n": 0}

        async def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("transient")
            return _embeddings_response([[0.1]])

        mock_client.embeddings.create_async = flaky

        provider = MistralEmbeddingProvider(_creds())
        response = await provider.aembed_query("hi", config=MistralEmbeddingConfig())
        assert calls["n"] == 2
        assert response.embeddings[0].to_list() == [0.1]
