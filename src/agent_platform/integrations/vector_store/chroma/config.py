from __future__ import annotations
from agent_platform.core.interfaces.vector_store.config import VectorStoreConfig


class ChromaConfig(VectorStoreConfig):
    # HTTP-client mode target; ignored when `persist_directory` is set.
    host: str = "localhost"
    port: int = 8000  # HTTP-client mode target port; ignored when `persist_directory` is set

    # When set, uses an embedded PersistentClient instead of the HTTP client.
    persist_directory: str | None = None

    # Forwarded to the HTTP-client path; HttpClient defaults to ssl=False.
    ssl: bool = False

    tenant: str = "default_tenant"  # Chroma multi-tenancy tenant name, forwarded to the client constructor
    database: str = "default_database"  # database within the tenant to connect to


"""
sources: https://docs.trychroma.com/reference/python/client
         https://docs.trychroma.com/production/administration/multi-tenancy
"""
