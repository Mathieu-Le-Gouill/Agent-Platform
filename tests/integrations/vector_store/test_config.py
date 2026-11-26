import pytest
from pydantic import ValidationError

from agent_platform.core.interfaces.vector_store.config import VectorStoreConfig
from agent_platform.integrations.vector_store.chroma.config import ChromaConfig
from agent_platform.integrations.vector_store.faiss.config import FAISSConfig
from agent_platform.integrations.vector_store.pinecone.config import PineconeConfig
from agent_platform.integrations.vector_store.qdrant.config import QdrantConfig
from agent_platform.integrations.vector_store.weaviate.config import WeaviateConfig


def test_vector_store_config_defaults():
    cfg = VectorStoreConfig(collection_name="test")
    assert cfg.collection_name == "test"
    assert cfg.namespace is None


def test_vector_store_config_custom():
    cfg = VectorStoreConfig(collection_name="docs", namespace="tenant-a")
    assert cfg.collection_name == "docs"
    assert cfg.namespace == "tenant-a"


@pytest.mark.parametrize(
    "config_cls",
    [
        VectorStoreConfig,
        ChromaConfig,
        QdrantConfig,
        PineconeConfig,
        WeaviateConfig,
        FAISSConfig,
    ],
)
def test_empty_namespace_is_rejected(config_cls):
    # "" is the shared default partition on Pinecone and Weaviate, so it
    # would silently give a tenant no isolation.
    with pytest.raises(ValidationError, match="namespace"):
        config_cls(namespace="")


def test_chroma_config_inherits():
    cfg = ChromaConfig(collection_name="chroma_test")
    assert isinstance(cfg, VectorStoreConfig)
    assert cfg.collection_name == "chroma_test"
    assert cfg.host == "localhost"
    assert cfg.port == 8000


def test_chroma_config_custom_host():
    cfg = ChromaConfig(collection_name="remote", host="10.0.0.1", port=9000)
    assert cfg.host == "10.0.0.1"
    assert cfg.port == 9000


def test_qdrant_config_inherits():
    cfg = QdrantConfig(collection_name="qdrant_test")
    assert isinstance(cfg, VectorStoreConfig)
    assert cfg.collection_name == "qdrant_test"
    assert cfg.url == "http://localhost:6333"


def test_qdrant_config_custom_url():
    cfg = QdrantConfig(collection_name="remote", url="https://qdrant.example.com")
    assert cfg.url == "https://qdrant.example.com"


def test_configs_are_frozen():
    cfg = ChromaConfig(collection_name="frozen")
    try:
        cfg.host = "other"  # type: ignore[misc]
        assert False, "expected ValidationError"
    except ValidationError:
        pass
