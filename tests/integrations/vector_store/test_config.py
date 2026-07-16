from pydantic import ValidationError

from agent_platform.core.interfaces.vector_store.config import (
    VectorStoreConfig,
    DistanceMetric,
)
from agent_platform.integrations.vector_store.chroma.config import ChromaConfig
from agent_platform.integrations.vector_store.qdrant.config import QdrantConfig


def test_distance_metric_enum():
    assert DistanceMetric.COSINE.value == "cosine"
    assert DistanceMetric.DOT.value == "dot"
    assert DistanceMetric.EUCLIDEAN.value == "euclidean"


def test_vector_store_config_defaults():
    cfg = VectorStoreConfig(collection_name="test", dimension=768)
    assert cfg.collection_name == "test"
    assert cfg.dimension == 768
    assert cfg.distance == DistanceMetric.COSINE
    assert cfg.top_k == 5


def test_vector_store_config_custom():
    cfg = VectorStoreConfig(
        collection_name="docs",
        dimension=384,
        distance=DistanceMetric.EUCLIDEAN,
        top_k=10,
    )
    assert cfg.collection_name == "docs"
    assert cfg.dimension == 384
    assert cfg.distance == DistanceMetric.EUCLIDEAN
    assert cfg.top_k == 10


def test_chroma_config_inherits():
    cfg = ChromaConfig(collection_name="chroma_test", dimension=768)
    assert isinstance(cfg, VectorStoreConfig)
    assert cfg.collection_name == "chroma_test"
    assert cfg.dimension == 768
    assert cfg.host == "localhost"
    assert cfg.port == 8000


def test_chroma_config_custom_host():
    cfg = ChromaConfig(
        collection_name="remote", dimension=512, host="10.0.0.1", port=9000
    )
    assert cfg.host == "10.0.0.1"
    assert cfg.port == 9000


def test_qdrant_config_inherits():
    cfg = QdrantConfig(collection_name="qdrant_test", dimension=128)
    assert isinstance(cfg, VectorStoreConfig)
    assert cfg.collection_name == "qdrant_test"
    assert cfg.dimension == 128
    assert cfg.url == "http://localhost:6333"


def test_qdrant_config_custom_url():
    cfg = QdrantConfig(
        collection_name="remote", dimension=256, url="https://qdrant.example.com"
    )
    assert cfg.url == "https://qdrant.example.com"


def test_configs_are_frozen():
    cfg = ChromaConfig(collection_name="frozen", dimension=64)
    try:
        cfg.host = "other"  # type: ignore[misc]
        assert False, "expected ValidationError"
    except ValidationError:
        pass
