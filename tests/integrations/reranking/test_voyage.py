from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from pydantic import SecretStr

from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.integrations.credentials import VoyageCredentials
from agent_platform.integrations.reranking.voyage.config import VoyageRerankerConfig
from agent_platform.integrations.reranking.voyage.provider import VoyageRerankerProvider


def _provider() -> VoyageRerankerProvider:
    return VoyageRerankerProvider(VoyageCredentials(api_key=SecretStr("test-key")))


def _result(index: int, score: float) -> MagicMock:
    result = MagicMock(spec=["index", "relevance_score"])
    result.index = index
    result.relevance_score = score
    return result


class TestClientKwargs:
    def test_default_max_retries(self):
        provider = _provider()
        kwargs = provider._client_kwargs(VoyageRerankerConfig(max_retries=None))
        assert kwargs["max_retries"] == 3
        assert "timeout" not in kwargs

    def test_explicit_timeout_and_retries(self):
        provider = _provider()
        cfg = VoyageRerankerConfig(timeout=15.0, max_retries=5)
        kwargs = provider._client_kwargs(cfg)
        assert kwargs["timeout"] == 15.0
        assert kwargs["max_retries"] == 5

    def test_client_options_fallback(self):
        from agent_platform.core.credentials import ClientOptions

        provider = VoyageRerankerProvider(
            VoyageCredentials(api_key=SecretStr("test-key")),
            client_options=ClientOptions(timeout=60.0, max_retries=7),
        )
        kwargs = provider._client_kwargs(VoyageRerankerConfig())
        assert kwargs["timeout"] == 60.0
        assert kwargs["max_retries"] == 7


class TestSync:
    def test_forwards_top_k_and_model(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.reranking.voyage.provider.voyageai.Client"
        )
        mock_client = MagicMock()
        mock_client.rerank.return_value = MagicMock(results=[])
        mock_client_cls.return_value = mock_client

        provider = _provider()
        provider.rerank(
            "q",
            [TextChunk(id=uuid4(), text="a", index=0)],
            VoyageRerankerConfig(top_k=3),
        )

        _, kwargs = mock_client.rerank.call_args
        assert kwargs["top_k"] == 3
        assert kwargs["model"] == "rerank-2.5"
        assert "truncation" not in kwargs

    def test_forwards_truncation_when_set(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.reranking.voyage.provider.voyageai.Client"
        )
        mock_client = MagicMock()
        mock_client.rerank.return_value = MagicMock(results=[])
        mock_client_cls.return_value = mock_client

        provider = _provider()
        provider.rerank(
            "q",
            [TextChunk(id=uuid4(), text="a", index=0)],
            VoyageRerankerConfig(truncation=False),
        )

        _, kwargs = mock_client.rerank.call_args
        assert kwargs["truncation"] is False

    def test_maps_results_by_index(self, mocker):
        items = [
            TextChunk(id=uuid4(), text="a", index=0),
            TextChunk(id=uuid4(), text="b", index=1),
        ]
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.reranking.voyage.provider.voyageai.Client"
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
    async def test_maps_results_with_scores(self, mocker):
        items = [TextChunk(id=uuid4(), text="a", index=0)]
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.reranking.voyage.provider.voyageai.AsyncClient"
        )
        mock_client = MagicMock()
        mock_client.rerank = AsyncMock(
            return_value=MagicMock(results=[_result(0, 0.42)])
        )
        mock_client_cls.return_value = mock_client

        provider = _provider()
        config = VoyageRerankerConfig(return_scores=True, normalize_scores=False)
        results = await provider.arerank("q", items, config)

        assert results[0].confidence.value == 0.42
