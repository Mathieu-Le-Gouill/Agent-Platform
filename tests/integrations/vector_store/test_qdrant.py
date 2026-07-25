import pytest

pytest.importorskip("qdrant_client")
pytest.importorskip("langchain_qdrant")

from agent_platform.integrations.credentials import QdrantCredentials
from agent_platform.integrations.vector_store.qdrant.config import QdrantConfig
from agent_platform.integrations.vector_store.qdrant.provider import (
    QdrantVectorStoreProvider,
)
from tests.helpers import assert_custom_construction_stored, assert_default_construction


@pytest.fixture
def provider():
    return QdrantVectorStoreProvider.__new__(QdrantVectorStoreProvider)


def test_qdrant_config_has_no_api_key_field():
    assert "api_key" not in QdrantConfig.model_fields


def test_qdrant_config_prefer_grpc_defaults_false():
    assert QdrantConfig().prefer_grpc is False


def test_qdrant_config_prefer_grpc_custom():
    assert QdrantConfig(prefer_grpc=True).prefer_grpc is True


class TestQdrantConstruction:
    def test_default_credentials_and_embeddings(self):
        assert_default_construction(QdrantVectorStoreProvider, QdrantConfig)

    def test_custom_credentials_and_embeddings_stored(self):
        assert_custom_construction_stored(
            QdrantVectorStoreProvider, QdrantCredentials(api_key="secret"), object()
        )


class TestQdrantBuildClient:
    def test_build_client_forwards_prefer_grpc(self, capture_client_kwargs):
        import agent_platform.integrations.vector_store.qdrant.provider as mod

        captured = capture_client_kwargs(mod, "QdrantClient", "QdrantVectorStore")

        provider = QdrantVectorStoreProvider.__new__(QdrantVectorStoreProvider)
        provider._credentials = QdrantCredentials(api_key=None)
        provider._embeddings = None

        provider._build_client(QdrantConfig(prefer_grpc=True))

        assert captured["prefer_grpc"] is True

    def test_build_client_still_uses_credentials_api_key(self, capture_client_kwargs):
        import agent_platform.integrations.vector_store.qdrant.provider as mod

        captured = capture_client_kwargs(mod, "QdrantClient", "QdrantVectorStore")

        provider = QdrantVectorStoreProvider.__new__(QdrantVectorStoreProvider)
        provider._credentials = QdrantCredentials(api_key="topsecret")
        provider._embeddings = None

        provider._build_client(QdrantConfig())

        assert captured["api_key"] == "topsecret"


class TestQdrantSearchKwargs:
    def test_no_filter_returns_none(self, provider):
        kwargs = provider._search_kwargs(QdrantConfig(), None)
        assert kwargs == {"filter": None}

    def test_empty_filter_returns_none(self, provider):
        kwargs = provider._search_kwargs(QdrantConfig(), {})
        assert kwargs == {"filter": None}

    def test_filter_builds_qdrant_filter_model(self, provider):
        from qdrant_client import models

        kwargs = provider._search_kwargs(QdrantConfig(), {"source": "doc.txt"})
        qfilter = kwargs["filter"]
        assert isinstance(qfilter, models.Filter)
        assert len(qfilter.must) == 1
        condition = qfilter.must[0]
        assert isinstance(condition, models.FieldCondition)
        assert condition.key == "source"
        assert condition.match == models.MatchValue(value="doc.txt")

    def test_multi_key_filter_produces_multiple_conditions(self, provider):
        kwargs = provider._search_kwargs(
            QdrantConfig(), {"source": "doc.txt", "language": "en"}
        )
        assert len(kwargs["filter"].must) == 2
