from pydantic import Field

from agent_platform.core.config import RequestOptions


class VectorStoreConfig(RequestOptions):
    collection_name: str = (
        "default"  # index/collection identifier to read and write vectors from
    )
    # Logical partition within a collection for multi-tenant isolation. Only
    # Pinecone (namespace) and Weaviate (tenant) honor it; Qdrant, Chroma and
    # FAISS raise ConfigError when it is set rather than mix tenants.
    # An empty string is rejected: it is the shared default partition on
    # Pinecone and Weaviate, so it would give a tenant no isolation.
    namespace: str | None = Field(default=None, min_length=1)
