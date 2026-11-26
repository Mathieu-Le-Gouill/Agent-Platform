from __future__ import annotations

from agent_platform.core.errors import ConfigError
from agent_platform.core.interfaces.vector_store.config import VectorStoreConfig


def reject_namespace(config: VectorStoreConfig, store: str) -> None:
    """Fail closed for a store that cannot isolate tenants by namespace.

    Raises `ConfigError` when `config.namespace` is set, so a tenant's data
    is never written to or read from the shared collection by accident.
    """
    if config.namespace is not None:
        raise ConfigError(
            f"{store} does not support namespace isolation; leave `namespace` "
            "unset, or use a store that honors it (Pinecone, Weaviate)"
        )
