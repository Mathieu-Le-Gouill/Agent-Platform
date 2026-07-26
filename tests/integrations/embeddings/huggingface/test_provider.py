from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from pydantic import SecretStr

from agent_platform.core.errors import MissingCredentialError, ProviderError
from agent_platform.core.interfaces.embeddings.response import EmbeddingResponse
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.integrations.credentials import HuggingFaceCredentials
from agent_platform.integrations.embeddings.huggingface.config import (
    HuggingFaceEmbeddingConfig,
    HuggingFaceEmbeddingMode,
)
from agent_platform.integrations.embeddings.huggingface.provider import (
    HuggingFaceEmbeddingProvider,
)


def _creds(key: str = "hf_test") -> HuggingFaceCredentials:
    return HuggingFaceCredentials(api_key=SecretStr(key))


class TestHuggingFaceEmbeddingProviderConstruction:
    def test_default_config_model(self):
        provider = HuggingFaceEmbeddingProvider()
        assert (
            provider._default_config().model == "sentence-transformers/all-MiniLM-L6-v2"
        )

    def test_default_config_mode(self):
        provider = HuggingFaceEmbeddingProvider()
        assert provider._default_config().mode == HuggingFaceEmbeddingMode.LOCAL

    def test_hosted_missing_api_key(self):
        provider = HuggingFaceEmbeddingProvider()
        provider._credentials = HuggingFaceCredentials()
        cfg = HuggingFaceEmbeddingConfig(mode=HuggingFaceEmbeddingMode.HOSTED)
        with pytest.raises(
            MissingCredentialError, match="Hugging Face Hub API token is required"
        ):
            provider._hosted_sync_client(cfg)


class TestHuggingFaceEmbeddingProviderLocalClient:
    def test_local_client_creation(self, mocker):
        mock_st = mocker.patch(
            "agent_platform.integrations.embeddings.huggingface.provider.SentenceTransformer"
        )
        provider = HuggingFaceEmbeddingProvider()
        cfg = HuggingFaceEmbeddingConfig(model_kwargs={"device": "cpu"})
        client = provider._local_client(cfg)
        assert client is mock_st.return_value
        mock_st.assert_called_once_with(cfg.model, device="cpu")

    def test_local_client_dimensions_maps_to_truncate_dim(self, mocker):
        mock_st = mocker.patch(
            "agent_platform.integrations.embeddings.huggingface.provider.SentenceTransformer"
        )
        provider = HuggingFaceEmbeddingProvider()
        provider._local_client(HuggingFaceEmbeddingConfig(dimensions=128))
        _, kwargs = mock_st.call_args
        assert kwargs["truncate_dim"] == 128

    def test_local_client_extra_params_applied(self, mocker):
        mock_st = mocker.patch(
            "agent_platform.integrations.embeddings.huggingface.provider.SentenceTransformer"
        )
        provider = HuggingFaceEmbeddingProvider()
        provider._local_client(
            HuggingFaceEmbeddingConfig(extra_params={"trust_remote_code": True})
        )
        _, kwargs = mock_st.call_args
        assert kwargs["trust_remote_code"] is True


class TestHuggingFaceEmbeddingProviderHostedClient:
    def test_hosted_client_kwargs(self):
        provider = HuggingFaceEmbeddingProvider(_creds())
        cfg = HuggingFaceEmbeddingConfig(
            mode=HuggingFaceEmbeddingMode.HOSTED, provider="cerebras", timeout=15.0
        )
        kwargs = provider._hosted_client_kwargs(cfg)
        assert kwargs["model"] == cfg.model
        assert kwargs["token"] == "hf_test"
        assert kwargs["provider"] == "cerebras"
        assert kwargs["timeout"] == 15.0

    def test_hosted_params(self):
        provider = HuggingFaceEmbeddingProvider(_creds())
        cfg = HuggingFaceEmbeddingConfig(
            dimensions=64, truncate=True, normalize=False, model_kwargs={"foo": "bar"}
        )
        params = provider._hosted_params(cfg)
        assert params["dimensions"] == 64
        assert params["truncate"] is True
        assert params["normalize"] is False
        assert params["foo"] == "bar"

    def test_hosted_params_empty_by_default(self):
        provider = HuggingFaceEmbeddingProvider(_creds())
        params = provider._hosted_params(HuggingFaceEmbeddingConfig())
        assert params == {}


class TestHuggingFaceEmbedDocumentLocal:
    def test_sync(self, mocker):
        mock_st_cls = mocker.patch(
            "agent_platform.integrations.embeddings.huggingface.provider.SentenceTransformer"
        )
        mock_st = MagicMock()
        mock_st_cls.return_value = mock_st
        mock_st.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]

        uid1, uid2 = uuid4(), uuid4()
        items = [TextChunk(id=uid1, text="hello"), TextChunk(id=uid2, text="world")]
        provider = HuggingFaceEmbeddingProvider()
        response = provider.embed_document(
            items,
            config=HuggingFaceEmbeddingConfig(mode=HuggingFaceEmbeddingMode.LOCAL),
        )

        assert isinstance(response, EmbeddingResponse)
        assert len(response) == 2
        assert response.embeddings[0].id == uid1
        assert response.embeddings[0].to_list() == [0.1, 0.2]
        mock_st.encode.assert_called_once_with(["hello", "world"])

    async def test_async(self, mocker):
        mock_st_cls = mocker.patch(
            "agent_platform.integrations.embeddings.huggingface.provider.SentenceTransformer"
        )
        mock_st = MagicMock()
        mock_st_cls.return_value = mock_st
        mock_st.encode.return_value = [[0.5, 0.6]]

        items = [TextChunk(id=uuid4(), text="single")]
        provider = HuggingFaceEmbeddingProvider()
        response = await provider.aembed_document(
            items,
            config=HuggingFaceEmbeddingConfig(mode=HuggingFaceEmbeddingMode.LOCAL),
        )

        assert len(response) == 1
        assert response.embeddings[0].to_list() == [0.5, 0.6]


class TestHuggingFaceEmbedDocumentHosted:
    def test_sync(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.embeddings.huggingface.provider.InferenceClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.feature_extraction.return_value = [[0.1, 0.2], [0.3, 0.4]]

        uid1, uid2 = uuid4(), uuid4()
        items = [TextChunk(id=uid1, text="hello"), TextChunk(id=uid2, text="world")]
        provider = HuggingFaceEmbeddingProvider(_creds())
        response = provider.embed_document(
            items,
            config=HuggingFaceEmbeddingConfig(mode=HuggingFaceEmbeddingMode.HOSTED),
        )

        assert len(response) == 2
        assert response.embeddings[0].id == uid1
        assert response.embeddings[0].to_list() == [0.1, 0.2]
        args, _ = mock_client.feature_extraction.call_args
        assert args[0] == ["hello", "world"]

    async def test_async(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.embeddings.huggingface.provider.AsyncInferenceClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.feature_extraction = AsyncMock(return_value=[[0.5, 0.6]])

        items = [TextChunk(id=uuid4(), text="single")]
        provider = HuggingFaceEmbeddingProvider(_creds())
        response = await provider.aembed_document(
            items,
            config=HuggingFaceEmbeddingConfig(mode=HuggingFaceEmbeddingMode.HOSTED),
        )

        assert len(response) == 1
        assert response.embeddings[0].to_list() == [0.5, 0.6]

    async def test_async_translates_permanent_failure(self, mocker, no_retry_sleep):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.embeddings.huggingface.provider.AsyncInferenceClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        async def always_fails(*args, **kwargs):
            raise ConnectionError("boom")

        mock_client.feature_extraction = always_fails

        provider = HuggingFaceEmbeddingProvider(_creds())
        with pytest.raises(ProviderError, match="Embedding generation failed"):
            await provider.aembed_document(
                [TextChunk(id=uuid4(), text="fail")],
                config=HuggingFaceEmbeddingConfig(mode=HuggingFaceEmbeddingMode.HOSTED),
            )


class TestHuggingFaceEmbedQuery:
    def test_local_sync(self, mocker):
        mock_st_cls = mocker.patch(
            "agent_platform.integrations.embeddings.huggingface.provider.SentenceTransformer"
        )
        mock_st = MagicMock()
        mock_st_cls.return_value = mock_st
        mock_st.encode.return_value = [[0.5, 0.6, 0.7]]

        provider = HuggingFaceEmbeddingProvider()
        response = provider.embed_query(
            "hello world",
            config=HuggingFaceEmbeddingConfig(mode=HuggingFaceEmbeddingMode.LOCAL),
        )

        assert len(response) == 1
        assert response.embeddings[0].to_list() == [0.5, 0.6, 0.7]
        mock_st.encode.assert_called_once_with(["hello world"])

    def test_hosted_sync(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.embeddings.huggingface.provider.InferenceClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.feature_extraction.return_value = [[0.8, 0.9]]

        provider = HuggingFaceEmbeddingProvider(_creds())
        response = provider.embed_query(
            "hi",
            config=HuggingFaceEmbeddingConfig(mode=HuggingFaceEmbeddingMode.HOSTED),
        )

        assert len(response) == 1
        assert response.embeddings[0].to_list() == [0.8, 0.9]

    async def test_hosted_async(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.embeddings.huggingface.provider.AsyncInferenceClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.feature_extraction = AsyncMock(return_value=[[1.0, 2.0]])

        provider = HuggingFaceEmbeddingProvider(_creds())
        response = await provider.aembed_query(
            "hi",
            config=HuggingFaceEmbeddingConfig(mode=HuggingFaceEmbeddingMode.HOSTED),
        )

        assert len(response) == 1
        assert response.embeddings[0].to_list() == [1.0, 2.0]

    async def test_local_async(self, mocker):
        mock_st_cls = mocker.patch(
            "agent_platform.integrations.embeddings.huggingface.provider.SentenceTransformer"
        )
        mock_st = MagicMock()
        mock_st_cls.return_value = mock_st
        mock_st.encode.return_value = [[0.1]]

        provider = HuggingFaceEmbeddingProvider()
        response = await provider.aembed_query(
            "hi",
            config=HuggingFaceEmbeddingConfig(mode=HuggingFaceEmbeddingMode.LOCAL),
        )

        assert response.embeddings[0].to_list() == [0.1]
