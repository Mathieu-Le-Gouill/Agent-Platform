from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from agent_platform.core.errors import ProviderError
from agent_platform.core.interfaces.embeddings.config import EmbeddingConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.integrations.embeddings.langchain_base import LangChainEmbedder


class _TestEmbedder(LangChainEmbedder):
    def _default_config(self):
        return EmbeddingConfig(model="test-model")

    def _client(self, config):
        raise NotImplementedError


class TestAembedDocumentRetryAndTranslation:
    async def test_retries_transient_failure_then_succeeds(self, no_retry_sleep):
        calls = {"n": 0}

        async def flaky(_texts):
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("transient")
            return [[0.1, 0.2]]

        mock_client = MagicMock()
        mock_client.aembed_documents = flaky

        embedder = _TestEmbedder()
        embedder._client = MagicMock(return_value=mock_client)

        chunk = TextChunk(id=uuid4(), text="hello", index=0)
        result = await embedder.aembed_document([chunk])

        assert calls["n"] == 2
        assert result.embeddings[0].to_list() == [0.1, 0.2]

    async def test_translates_permanent_failure_to_provider_error(self, no_retry_sleep):
        async def always_fails(_texts):
            raise ConnectionError("boom")

        mock_client = MagicMock()
        mock_client.aembed_documents = always_fails

        embedder = _TestEmbedder()
        embedder._client = MagicMock(return_value=mock_client)

        chunk = TextChunk(id=uuid4(), text="hello", index=0)
        with pytest.raises(ProviderError, match="Embedding generation failed"):
            await embedder.aembed_document([chunk])


class TestAembedQueryRetryAndTranslation:
    async def test_retries_transient_failure_then_succeeds(self, no_retry_sleep):
        calls = {"n": 0}

        async def flaky(_query):
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("transient")
            return [0.3, 0.4]

        mock_client = MagicMock()
        mock_client.aembed_query = flaky

        embedder = _TestEmbedder()
        embedder._client = MagicMock(return_value=mock_client)

        result = await embedder.aembed_query("hi")

        assert calls["n"] == 2
        assert result.embeddings[0].to_list() == [0.3, 0.4]

    async def test_translates_permanent_failure_to_provider_error(self, no_retry_sleep):
        async def always_fails(_query):
            raise ConnectionError("boom")

        mock_client = MagicMock()
        mock_client.aembed_query = always_fails

        embedder = _TestEmbedder()
        embedder._client = MagicMock(return_value=mock_client)

        with pytest.raises(ProviderError, match="Embedding generation failed"):
            await embedder.aembed_query("hi")
