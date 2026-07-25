import pytest

pytest.importorskip("langchain_pinecone")

from agent_platform.core.errors import MissingCredentialError
from agent_platform.integrations.credentials import PineconeCredentials
from agent_platform.integrations.vector_store.pinecone.config import PineconeConfig
from agent_platform.integrations.vector_store.pinecone.provider import PineconeStore
from tests.helpers import assert_custom_construction_stored, assert_default_construction


def test_pinecone_config_has_no_environment_field():
    assert "environment" not in PineconeConfig.model_fields


def test_pinecone_config_host_defaults_to_none():
    cfg = PineconeConfig()
    assert cfg.host is None


def test_pinecone_config_host_custom():
    cfg = PineconeConfig(host="my-index-abc123.svc.us-east-1.pinecone.io")
    assert cfg.host == "my-index-abc123.svc.us-east-1.pinecone.io"


class TestPineconeConstruction:
    def test_default_credentials_and_embeddings(self):
        assert_default_construction(PineconeStore, PineconeConfig)

    def test_custom_credentials_and_embeddings_stored(self):
        assert_custom_construction_stored(
            PineconeStore, PineconeCredentials(api_key="secret"), object()
        )


class TestPineconeBuildClient:
    def test_build_client_raises_without_api_key(self):
        store = PineconeStore.__new__(PineconeStore)
        store._credentials = PineconeCredentials(api_key=None)
        store._embeddings = None

        with pytest.raises(MissingCredentialError):
            store._build_client(PineconeConfig())

    def test_build_client_omits_host_when_unset(self, capture_client_kwargs):
        import agent_platform.integrations.vector_store.pinecone.provider as mod

        captured = capture_client_kwargs(mod, "PineconeVectorStore")

        store = PineconeStore.__new__(PineconeStore)
        store._credentials = mod.PineconeCredentials(api_key="secret")
        store._embeddings = None

        store._build_client(PineconeConfig())

        assert "host" not in captured

    def test_build_client_forwards_host_when_set(self, capture_client_kwargs):
        import agent_platform.integrations.vector_store.pinecone.provider as mod

        captured = capture_client_kwargs(mod, "PineconeVectorStore")

        store = PineconeStore.__new__(PineconeStore)
        store._credentials = mod.PineconeCredentials(api_key="secret")
        store._embeddings = None

        store._build_client(PineconeConfig(host="idx.svc.pinecone.io"))

        assert captured["host"] == "idx.svc.pinecone.io"
