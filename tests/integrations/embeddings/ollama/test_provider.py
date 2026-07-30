from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from agent_platform.core.credentials import ClientOptions
from agent_platform.core.errors import ProviderError
from agent_platform.core.interfaces.embeddings.response import EmbeddingResponse
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.integrations.credentials import OllamaCredentials
from agent_platform.integrations.embeddings.ollama.config import OllamaEmbeddingConfig
from agent_platform.integrations.embeddings.ollama.provider import (
    OllamaEmbeddingProvider,
)


def _embed_response(vectors: list[list[float]]) -> SimpleNamespace:
    return SimpleNamespace(embeddings=vectors)


class TestOllamaEmbeddingProviderConstruction:
    def test_default_config_model(self):
        provider = OllamaEmbeddingProvider()
        assert provider._default_config().model == "nomic-embed-text"

    def test_no_credential_needed(self):
        provider = OllamaEmbeddingProvider(OllamaCredentials())
        client = provider._client(OllamaEmbeddingConfig())
        assert client is not None


class TestOllamaEmbeddingProviderClient:
    def test_client_kwargs_default_no_timeout(self):
        provider = OllamaEmbeddingProvider()
        kwargs = provider._client_kwargs(OllamaEmbeddingConfig())
        assert "timeout" not in kwargs

    def test_client_kwargs_default_host_falls_back_to_localhost(self):
        provider = OllamaEmbeddingProvider()
        kwargs = provider._client_kwargs(OllamaEmbeddingConfig())
        assert kwargs["host"] == "http://localhost:11434"

    def test_client_kwargs_explicit_host(self):
        provider = OllamaEmbeddingProvider(
            client_options=ClientOptions(base_url="http://custom-ollama:1234")
        )
        kwargs = provider._client_kwargs(OllamaEmbeddingConfig())
        assert kwargs["host"] == "http://custom-ollama:1234"

    def test_client_kwargs_explicit_timeout(self):
        provider = OllamaEmbeddingProvider()
        kwargs = provider._client_kwargs(OllamaEmbeddingConfig(timeout=25.0))
        assert kwargs["timeout"] == 25.0

    def test_params_dimensions_included(self):
        provider = OllamaEmbeddingProvider()
        params = provider._params(OllamaEmbeddingConfig(dimensions=384))
        assert params["dimensions"] == 384

    def test_params_dimensions_omitted_when_none(self):
        provider = OllamaEmbeddingProvider()
        params = provider._params(OllamaEmbeddingConfig())
        assert "dimensions" not in params

    def test_params_options(self):
        provider = OllamaEmbeddingProvider()
        cfg = OllamaEmbeddingConfig(temperature=0.3, top_p=0.9, top_k=40)
        params = provider._params(cfg)
        assert params["options"] == {"temperature": 0.3, "top_p": 0.9, "top_k": 40}

    def test_params_options_omitted_when_unset(self):
        provider = OllamaEmbeddingProvider()
        params = provider._params(OllamaEmbeddingConfig())
        assert "options" not in params

    def test_params_keep_alive_included_when_set(self):
        provider = OllamaEmbeddingProvider()
        params = provider._params(OllamaEmbeddingConfig(keep_alive=600))
        assert params["keep_alive"] == 600

    def test_params_keep_alive_accepts_duration_string(self):
        provider = OllamaEmbeddingProvider()
        params = provider._params(OllamaEmbeddingConfig(keep_alive="5m"))
        assert params["keep_alive"] == "5m"

    def test_params_keep_alive_omitted_when_none(self):
        provider = OllamaEmbeddingProvider()
        params = provider._params(OllamaEmbeddingConfig())
        assert "keep_alive" not in params

    def test_params_extra_params_applied(self):
        provider = OllamaEmbeddingProvider()
        params = provider._params(
            OllamaEmbeddingConfig(extra_params={"truncate": True})
        )
        assert params["truncate"] is True


class TestOllamaEmbedDocument:
    def test_sync(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.embeddings.ollama.provider.Client"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.embed.return_value = _embed_response([[0.1, 0.2], [0.3, 0.4]])

        uid1, uid2 = uuid4(), uuid4()
        items = [TextChunk(id=uid1, text="hello"), TextChunk(id=uid2, text="world")]
        provider = OllamaEmbeddingProvider()
        response = provider.embed_document(items, config=OllamaEmbeddingConfig())

        assert isinstance(response, EmbeddingResponse)
        assert len(response) == 2
        assert response.embeddings[0].id == uid1
        assert response.embeddings[0].to_list() == [0.1, 0.2]
        _, kwargs = mock_client.embed.call_args
        assert kwargs["input"] == ["hello", "world"]

    def test_empty_list(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.embeddings.ollama.provider.Client"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.embed.return_value = _embed_response([])

        provider = OllamaEmbeddingProvider()
        response = provider.embed_document([], config=OllamaEmbeddingConfig())
        assert len(response) == 0

    async def test_async(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.embeddings.ollama.provider.AsyncClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.embed = AsyncMock(return_value=_embed_response([[0.5, 0.6]]))

        items = [TextChunk(id=uuid4(), text="single")]
        provider = OllamaEmbeddingProvider()
        response = await provider.aembed_document(items, config=OllamaEmbeddingConfig())

        assert len(response) == 1
        assert response.embeddings[0].to_list() == [0.5, 0.6]

    async def test_async_translates_permanent_failure(self, mocker, no_retry_sleep):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.embeddings.ollama.provider.AsyncClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        async def always_fails(*args, **kwargs):
            raise ConnectionError("boom")

        mock_client.embed = always_fails

        provider = OllamaEmbeddingProvider()
        with pytest.raises(ProviderError, match="Embedding generation failed"):
            await provider.aembed_document(
                [TextChunk(id=uuid4(), text="fail")], config=OllamaEmbeddingConfig()
            )


class TestOllamaEmbedQuery:
    def test_sync(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.embeddings.ollama.provider.Client"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.embed.return_value = _embed_response([[0.5, 0.6, 0.7]])

        provider = OllamaEmbeddingProvider()
        response = provider.embed_query("hello world", config=OllamaEmbeddingConfig())

        assert len(response) == 1
        assert response.embeddings[0].to_list() == [0.5, 0.6, 0.7]
        _, kwargs = mock_client.embed.call_args
        assert kwargs["input"] == ["hello world"]

    async def test_async(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.embeddings.ollama.provider.AsyncClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.embed = AsyncMock(return_value=_embed_response([[0.8, 0.9]]))

        provider = OllamaEmbeddingProvider()
        response = await provider.aembed_query(
            "test query", config=OllamaEmbeddingConfig()
        )

        assert len(response) == 1
        assert response.embeddings[0].to_list() == [0.8, 0.9]

    async def test_async_retries_transient_failure_then_succeeds(
        self, mocker, no_retry_sleep
    ):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.embeddings.ollama.provider.AsyncClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        calls = {"n": 0}

        async def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("transient")
            return _embed_response([[0.1]])

        mock_client.embed = flaky

        provider = OllamaEmbeddingProvider()
        response = await provider.aembed_query("hi", config=OllamaEmbeddingConfig())
        assert calls["n"] == 2
        assert response.embeddings[0].to_list() == [0.1]
