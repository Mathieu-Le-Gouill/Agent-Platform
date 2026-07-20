from pydantic import SecretStr

from agent_platform.integrations.credentials import VoyageCredentials
from agent_platform.integrations.reranking.voyage.voyage import VoyageRerankerProvider
from agent_platform.integrations.reranking.voyage.config import VoyageRerankerConfig


def _provider() -> VoyageRerankerProvider:
    return VoyageRerankerProvider(VoyageCredentials(api_key=SecretStr("test-key")))


def test_client_forwards_top_k():
    provider = _provider()
    client = provider._client(VoyageRerankerConfig(top_k=3))
    assert client.top_k == 3


def test_client_forwards_truncation_when_set():
    provider = _provider()
    client = provider._client(VoyageRerankerConfig(truncation=False))
    assert client.truncation is False


def test_client_truncation_defaults_to_library_default_when_unset():
    provider = _provider()
    client = provider._client(VoyageRerankerConfig())
    assert client.truncation is True


def test_client_uses_current_default_model():
    provider = _provider()
    client = provider._client(VoyageRerankerConfig())
    assert client.model == "rerank-2.5"
