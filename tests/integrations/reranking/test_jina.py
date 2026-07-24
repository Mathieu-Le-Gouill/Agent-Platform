from pydantic import SecretStr

from agent_platform.integrations.credentials import JinaCredentials
from agent_platform.integrations.reranking.jina.config import JinaRerankerConfig
from agent_platform.integrations.reranking.jina.jina import JinaRerankerProvider


def _provider() -> JinaRerankerProvider:
    return JinaRerankerProvider(JinaCredentials(api_key=SecretStr("test-key")))


def test_client_forwards_top_k_as_top_n():
    provider = _provider()
    client = provider._client(JinaRerankerConfig(top_k=7))
    assert client.top_n == 7


def test_client_top_n_none_when_top_k_unset():
    provider = _provider()
    client = provider._client(JinaRerankerConfig())
    assert client.top_n is None
