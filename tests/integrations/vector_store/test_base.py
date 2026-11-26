import pytest

from agent_platform.core.errors import ConfigError
from agent_platform.core.interfaces.vector_store.config import VectorStoreConfig
from agent_platform.integrations.vector_store._base import reject_namespace


def test_reject_namespace_passes_without_namespace():
    reject_namespace(VectorStoreConfig(), "Qdrant")


def test_reject_namespace_raises_config_error_naming_the_store():
    with pytest.raises(
        ConfigError, match="Qdrant does not support namespace isolation"
    ) as exc_info:
        reject_namespace(VectorStoreConfig(namespace="tenant-a"), "Qdrant")

    assert "tenant-a" not in str(exc_info.value)


def test_reject_namespace_treats_empty_string_as_set():
    with pytest.raises(ConfigError):
        reject_namespace(VectorStoreConfig(namespace=""), "Chroma")
