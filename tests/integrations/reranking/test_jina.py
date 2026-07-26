from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from pydantic import SecretStr

from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.integrations.credentials import JinaCredentials
from agent_platform.integrations.reranking.jina.config import JinaRerankerConfig
from agent_platform.integrations.reranking.jina.provider import JinaRerankerProvider


def _provider() -> JinaRerankerProvider:
    return JinaRerankerProvider(JinaCredentials(api_key=SecretStr("test-key")))


def _mock_response(results: list[dict]) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {"results": results}
    response.raise_for_status.return_value = None
    return response


class TestHeaders:
    def test_includes_bearer_token_when_api_key_set(self):
        headers = _provider()._headers()
        assert headers["Authorization"] == "Bearer test-key"

    def test_no_authorization_header_when_api_key_unset(self):
        provider = JinaRerankerProvider(JinaCredentials(api_key=None))
        assert "Authorization" not in provider._headers()


class TestSync:
    def test_forwards_top_k_as_top_n(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.reranking.jina.provider.httpx.Client"
        )
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response([])
        mock_client.__enter__.return_value = mock_client
        mock_client_cls.return_value = mock_client

        provider = _provider()
        provider.rerank(
            "q", [TextChunk(id=uuid4(), text="a", index=0)], JinaRerankerConfig(top_k=5)
        )

        _, kwargs = mock_client.post.call_args
        assert kwargs["json"]["top_n"] == 5
        assert kwargs["json"]["model"] == "jina-reranker-v3"

    def test_top_n_omitted_when_top_k_unset(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.reranking.jina.provider.httpx.Client"
        )
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response([])
        mock_client.__enter__.return_value = mock_client
        mock_client_cls.return_value = mock_client

        provider = _provider()
        provider.rerank("q", [TextChunk(id=uuid4(), text="a", index=0)])

        _, kwargs = mock_client.post.call_args
        assert "top_n" not in kwargs["json"]

    def test_maps_results_by_index(self, mocker):
        items = [
            TextChunk(id=uuid4(), text="a", index=0),
            TextChunk(id=uuid4(), text="b", index=1),
        ]
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.reranking.jina.provider.httpx.Client"
        )
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response(
            [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]
        )
        mock_client.__enter__.return_value = mock_client
        mock_client_cls.return_value = mock_client

        provider = _provider()
        results = provider.rerank("q", items)

        assert [c.text for c in results] == ["b", "a"]


class TestAsync:
    async def test_maps_results_with_scores(self, mocker):
        items = [TextChunk(id=uuid4(), text="a", index=0)]
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.reranking.jina.provider.httpx.AsyncClient"
        )
        mock_client = MagicMock()
        mock_client.post = AsyncMock(
            return_value=_mock_response([{"index": 0, "relevance_score": 0.42}])
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        provider = _provider()
        config = JinaRerankerConfig(return_scores=True, normalize_scores=False)
        results = await provider.arerank("q", items, config)

        assert results[0].confidence.value == 0.42
