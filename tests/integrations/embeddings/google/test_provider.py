from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

pytest.importorskip("google.genai")

from agent_platform.core.errors import MissingCredentialError, ProviderError
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.integrations.credentials import GoogleCredentials
from agent_platform.integrations.embeddings.google.config import GoogleEmbeddingConfig
from agent_platform.integrations.embeddings.google.provider import (
    GoogleEmbeddingProvider,
)
from tests.helpers import assert_custom_construction_stored, assert_default_construction


def _creds(key: str = "gm-test") -> GoogleCredentials:
    return GoogleCredentials(api_key=SecretStr(key))


def _response(vectors: list[list[float]]) -> SimpleNamespace:
    return SimpleNamespace(embeddings=[SimpleNamespace(values=v) for v in vectors])


class TestGoogleEmbeddingConstruction:
    def test_default_credentials_and_config(self):
        assert_default_construction(GoogleEmbeddingProvider, GoogleEmbeddingConfig)

    def test_custom_credentials_stored(self):
        assert_custom_construction_stored(GoogleEmbeddingProvider, _creds())

    def test_missing_api_key_raises(self):
        provider = GoogleEmbeddingProvider(GoogleCredentials(api_key=None))
        with pytest.raises(MissingCredentialError):
            provider._client()


class TestGoogleEmbedSync:
    def test_embed_sync_returns_vectors(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.embeddings.google.provider.genai.Client"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.embed_content.return_value = _response([[0.1, 0.2]])

        provider = GoogleEmbeddingProvider(_creds())
        result = provider.embed_document([TextChunk(text="hi", index=0)])

        assert result.embeddings[0].vector == (0.1, 0.2)

    def test_embed_sync_forwards_dimensions(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.embeddings.google.provider.genai.Client"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.embed_content.return_value = _response([[0.1]])

        provider = GoogleEmbeddingProvider(_creds())
        config = GoogleEmbeddingConfig(dimensions=256)
        provider.embed_query("hi", config=config)

        _, kwargs = mock_client.models.embed_content.call_args
        assert kwargs["config"].output_dimensionality == 256

    def test_embed_sync_raises_on_missing_vector(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.embeddings.google.provider.genai.Client"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.embed_content.return_value = SimpleNamespace(
            embeddings=[SimpleNamespace(values=None)]
        )

        provider = GoogleEmbeddingProvider(_creds())
        with pytest.raises(ProviderError, match="no embedding vector"):
            provider.embed_query("hi")


class TestGoogleEmbedAsync:
    async def test_aembed_query(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.embeddings.google.provider.genai.Client"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.aio.models.embed_content = AsyncMock(
            return_value=_response([[0.3, 0.4]])
        )

        provider = GoogleEmbeddingProvider(_creds())
        result = await provider.aembed_query("hi")

        assert result.embeddings[0].vector == (0.3, 0.4)
