from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from pydantic import SecretStr

from agent_platform.core.errors import MissingCredentialError
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.integrations.credentials import CohereCredentials
from agent_platform.integrations.reranking.cohere.config import CohereRerankerConfig
from agent_platform.integrations.reranking.cohere.provider import CohereRerankerProvider


def _provider() -> CohereRerankerProvider:
    return CohereRerankerProvider(CohereCredentials(api_key=SecretStr("test-key")))


def _result(index: int, score: float) -> MagicMock:
    result = MagicMock(spec=["index", "relevance_score"])
    result.index = index
    result.relevance_score = score
    return result


def test_missing_credentials_raises_on_sync_client():
    provider = CohereRerankerProvider(CohereCredentials(api_key=None))
    with pytest.raises(MissingCredentialError):
        provider._sync_client(CohereRerankerConfig())


class TestClientKwargs:
    def test_default_max_retries(self):
        provider = _provider()
        kwargs = provider._client_kwargs(CohereRerankerConfig(max_retries=None))
        assert kwargs["max_retries"] == 3
        assert "timeout" not in kwargs

    def test_explicit_timeout_and_retries(self):
        provider = _provider()
        cfg = CohereRerankerConfig(timeout=15.0, max_retries=5)
        kwargs = provider._client_kwargs(cfg)
        assert kwargs["timeout"] == 15.0
        assert kwargs["max_retries"] == 5

    def test_client_options_fallback(self):
        from agent_platform.core.credentials import ClientOptions

        provider = CohereRerankerProvider(
            CohereCredentials(api_key=SecretStr("test-key")),
            client_options=ClientOptions(timeout=60.0, max_retries=7),
        )
        kwargs = provider._client_kwargs(CohereRerankerConfig())
        assert kwargs["timeout"] == 60.0
        assert kwargs["max_retries"] == 7


class TestSync:
    def test_forwards_top_k_as_top_n(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.reranking.cohere.provider.cohere.Client"
        )
        mock_client = MagicMock()
        mock_client.rerank.return_value = MagicMock(results=[])
        mock_client_cls.return_value = mock_client

        provider = _provider()
        provider.rerank(
            "q",
            [TextChunk(id=uuid4(), text="a", index=0)],
            CohereRerankerConfig(top_k=5),
        )

        _, kwargs = mock_client.rerank.call_args
        assert kwargs["top_n"] == 5
        assert kwargs["model"] == "rerank-v4.0-fast"

    def test_max_chunks_per_doc_forwarded_when_set(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.reranking.cohere.provider.cohere.Client"
        )
        mock_client = MagicMock()
        mock_client.rerank.return_value = MagicMock(results=[])
        mock_client_cls.return_value = mock_client

        provider = _provider()
        provider.rerank(
            "q",
            [TextChunk(id=uuid4(), text="a", index=0)],
            CohereRerankerConfig(max_chunks_per_doc=5),
        )

        _, kwargs = mock_client.rerank.call_args
        assert kwargs["max_chunks_per_doc"] == 5

    def test_max_chunks_per_doc_omitted_when_unset(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.reranking.cohere.provider.cohere.Client"
        )
        mock_client = MagicMock()
        mock_client.rerank.return_value = MagicMock(results=[])
        mock_client_cls.return_value = mock_client

        provider = _provider()
        provider.rerank("q", [TextChunk(id=uuid4(), text="a", index=0)])

        _, kwargs = mock_client.rerank.call_args
        assert "max_chunks_per_doc" not in kwargs

    def test_maps_results_by_index(self, mocker):
        items = [
            TextChunk(id=uuid4(), text="a", index=0),
            TextChunk(id=uuid4(), text="b", index=1),
        ]
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.reranking.cohere.provider.cohere.Client"
        )
        mock_client = MagicMock()
        mock_client.rerank.return_value = MagicMock(
            results=[_result(1, 0.9), _result(0, 0.1)]
        )
        mock_client_cls.return_value = mock_client

        provider = _provider()
        results = provider.rerank("q", items)

        assert [c.text for c in results] == ["b", "a"]


class TestAsync:
    async def test_forwards_top_k_as_top_n(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.reranking.cohere.provider.cohere.AsyncClient"
        )
        mock_client = MagicMock()
        mock_client.rerank = AsyncMock(return_value=MagicMock(results=[]))
        mock_client_cls.return_value = mock_client

        provider = _provider()
        await provider.arerank(
            "q",
            [TextChunk(id=uuid4(), text="a", index=0)],
            CohereRerankerConfig(top_k=5),
        )

        _, kwargs = mock_client.rerank.await_args
        assert kwargs["top_n"] == 5

    async def test_maps_results_with_scores(self, mocker):
        items = [TextChunk(id=uuid4(), text="a", index=0)]
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.reranking.cohere.provider.cohere.AsyncClient"
        )
        mock_client = MagicMock()
        mock_client.rerank = AsyncMock(
            return_value=MagicMock(results=[_result(0, 0.42)])
        )
        mock_client_cls.return_value = mock_client

        provider = _provider()
        config = CohereRerankerConfig(return_scores=True, normalize_scores=False)
        results = await provider.arerank("q", items, config)

        assert results[0].confidence.value == 0.42
