from pydantic import SecretStr

from agent_platform.integrations.credentials import CohereCredentials
from agent_platform.integrations.reranking.cohere.config import CohereRerankerConfig
from agent_platform.integrations.reranking.cohere.provider import CohereRerankerProvider


def _provider() -> CohereRerankerProvider:
    return CohereRerankerProvider(CohereCredentials(api_key=SecretStr("test-key")))


def test_client_forwards_top_k_as_top_n():
    provider = _provider()
    client = provider._client(CohereRerankerConfig(top_k=5))
    assert client.top_n == 5


def test_client_top_n_none_when_top_k_unset():
    provider = _provider()
    client = provider._client(CohereRerankerConfig())
    assert client.top_n is None


def test_client_uses_configured_model():
    provider = _provider()
    client = provider._client(CohereRerankerConfig(model="rerank-v3.5"))
    assert client.model == "rerank-v3.5"
