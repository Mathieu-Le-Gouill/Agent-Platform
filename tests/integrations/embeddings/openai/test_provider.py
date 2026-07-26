from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from pydantic import SecretStr

from agent_platform.core.errors import MissingCredentialError, ProviderError
from agent_platform.core.interfaces.embeddings.response import EmbeddingResponse
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.integrations.credentials import OpenAICredentials
from agent_platform.integrations.embeddings.openai.config import OpenAIEmbeddingConfig
from agent_platform.integrations.embeddings.openai.provider import (
    OpenAIEmbeddingProvider,
)


def _creds(key: str = "sk-test") -> OpenAICredentials:
    return OpenAICredentials(api_key=SecretStr(key))


def _embeddings_response(vectors: list[list[float]]) -> SimpleNamespace:
    return SimpleNamespace(
        data=[SimpleNamespace(embedding=v, index=i) for i, v in enumerate(vectors)]
    )


class TestOpenAIEmbeddingProviderConstruction:
    def test_default_config_model(self):
        provider = OpenAIEmbeddingProvider()
        assert provider._default_config().model == "text-embedding-ada-002"

    def test_missing_api_key(self):
        provider = OpenAIEmbeddingProvider()
        provider._credentials = OpenAICredentials()
        cfg = OpenAIEmbeddingConfig(model="text-embedding-ada-002")
        with pytest.raises(MissingCredentialError, match="OpenAI API key is required"):
            provider._client(cfg)


class TestOpenAIEmbeddingProviderClient:
    def test_client_kwargs_defaults(self):
        provider = OpenAIEmbeddingProvider(_creds())
        kwargs = provider._client_kwargs(OpenAIEmbeddingConfig())
        assert kwargs["api_key"] == "sk-test"
        assert kwargs["max_retries"] == 3
        assert "timeout" not in kwargs

    def test_client_kwargs_explicit_timeout_and_retries(self):
        provider = OpenAIEmbeddingProvider(_creds())
        cfg = OpenAIEmbeddingConfig(timeout=15.0, max_retries=5)
        kwargs = provider._client_kwargs(cfg)
        assert kwargs["timeout"] == 15.0
        assert kwargs["max_retries"] == 5

    def test_client_kwargs_organization_included(self):
        creds = OpenAICredentials(api_key=SecretStr("sk-test"), organization="my-org")
        provider = OpenAIEmbeddingProvider(creds)
        kwargs = provider._client_kwargs(OpenAIEmbeddingConfig())
        assert kwargs["organization"] == "my-org"

    def test_client_kwargs_organization_omitted_when_none(self):
        provider = OpenAIEmbeddingProvider(_creds())
        kwargs = provider._client_kwargs(OpenAIEmbeddingConfig())
        assert "organization" not in kwargs

    def test_params_dimensions_included(self):
        provider = OpenAIEmbeddingProvider(_creds())
        params = provider._params(OpenAIEmbeddingConfig(dimensions=256))
        assert params["dimensions"] == 256

    def test_params_dimensions_omitted_when_none(self):
        provider = OpenAIEmbeddingProvider(_creds())
        params = provider._params(OpenAIEmbeddingConfig())
        assert "dimensions" not in params

    def test_params_encoding_format_included(self):
        provider = OpenAIEmbeddingProvider(_creds())
        params = provider._params(OpenAIEmbeddingConfig(encoding_format="base64"))
        assert params["encoding_format"] == "base64"

    def test_params_model_kwargs_merged(self):
        provider = OpenAIEmbeddingProvider(_creds())
        params = provider._params(OpenAIEmbeddingConfig(model_kwargs={"user": "u1"}))
        assert params["user"] == "u1"

    def test_params_extra_params_applied(self):
        provider = OpenAIEmbeddingProvider(_creds())
        params = provider._params(OpenAIEmbeddingConfig(extra_params={"foo": "bar"}))
        assert params["foo"] == "bar"


class TestOpenAIEmbedDocument:
    def test_sync(self, mocker):
        mock_openai = mocker.patch(
            "agent_platform.integrations.embeddings.openai.provider.OpenAI"
        )
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.embeddings.create.return_value = _embeddings_response(
            [[0.1, 0.2], [0.3, 0.4]]
        )

        uid1, uid2 = uuid4(), uuid4()
        items = [TextChunk(id=uid1, text="hello"), TextChunk(id=uid2, text="world")]
        provider = OpenAIEmbeddingProvider(_creds())
        response = provider.embed_document(items, config=OpenAIEmbeddingConfig())

        assert isinstance(response, EmbeddingResponse)
        assert len(response) == 2
        assert response.embeddings[0].id == uid1
        assert response.embeddings[0].to_list() == [0.1, 0.2]
        _, kwargs = mock_client.embeddings.create.call_args
        assert kwargs["input"] == ["hello", "world"]

    def test_empty_list(self, mocker):
        mock_openai = mocker.patch(
            "agent_platform.integrations.embeddings.openai.provider.OpenAI"
        )
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.embeddings.create.return_value = _embeddings_response([])

        provider = OpenAIEmbeddingProvider(_creds())
        response = provider.embed_document([], config=OpenAIEmbeddingConfig())
        assert len(response) == 0

    async def test_async(self, mocker):
        mock_openai = mocker.patch(
            "agent_platform.integrations.embeddings.openai.provider.AsyncOpenAI"
        )
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.embeddings.create = AsyncMock(
            return_value=_embeddings_response([[0.5, 0.6]])
        )

        uid = uuid4()
        items = [TextChunk(id=uid, text="single")]
        provider = OpenAIEmbeddingProvider(_creds())
        response = await provider.aembed_document(items, config=OpenAIEmbeddingConfig())

        assert len(response) == 1
        assert response.embeddings[0].to_list() == [0.5, 0.6]

    async def test_async_translates_permanent_failure(self, mocker, no_retry_sleep):
        mock_openai = mocker.patch(
            "agent_platform.integrations.embeddings.openai.provider.AsyncOpenAI"
        )
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        async def always_fails(*args, **kwargs):
            raise ConnectionError("boom")

        mock_client.embeddings.create = always_fails

        provider = OpenAIEmbeddingProvider(_creds())
        with pytest.raises(ProviderError, match="Embedding generation failed"):
            await provider.aembed_document(
                [TextChunk(id=uuid4(), text="fail")],
                config=OpenAIEmbeddingConfig(),
            )


class TestOpenAIEmbedQuery:
    def test_sync(self, mocker):
        mock_openai = mocker.patch(
            "agent_platform.integrations.embeddings.openai.provider.OpenAI"
        )
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.embeddings.create.return_value = _embeddings_response(
            [[0.5, 0.6, 0.7]]
        )

        provider = OpenAIEmbeddingProvider(_creds())
        response = provider.embed_query("hello world", config=OpenAIEmbeddingConfig())

        assert len(response) == 1
        assert response.embeddings[0].to_list() == [0.5, 0.6, 0.7]
        _, kwargs = mock_client.embeddings.create.call_args
        assert kwargs["input"] == ["hello world"]

    async def test_async(self, mocker):
        mock_openai = mocker.patch(
            "agent_platform.integrations.embeddings.openai.provider.AsyncOpenAI"
        )
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.embeddings.create = AsyncMock(
            return_value=_embeddings_response([[0.8, 0.9]])
        )

        provider = OpenAIEmbeddingProvider(_creds())
        response = await provider.aembed_query(
            "test query", config=OpenAIEmbeddingConfig()
        )

        assert len(response) == 1
        assert response.embeddings[0].to_list() == [0.8, 0.9]

    async def test_async_retries_transient_failure_then_succeeds(
        self, mocker, no_retry_sleep
    ):
        mock_openai = mocker.patch(
            "agent_platform.integrations.embeddings.openai.provider.AsyncOpenAI"
        )
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        calls = {"n": 0}

        async def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("transient")
            return _embeddings_response([[0.1]])

        mock_client.embeddings.create = flaky

        provider = OpenAIEmbeddingProvider(_creds())
        response = await provider.aembed_query("hi", config=OpenAIEmbeddingConfig())
        assert calls["n"] == 2
        assert response.embeddings[0].to_list() == [0.1]
