from agent_platform.core.config import RequestOptions


class VectorStoreConfig(RequestOptions):
    collection_name: str = (
        "default"  # index/collection identifier to read and write vectors from
    )
    # Logical partition within a collection for multi-tenant isolation. Only
    # Pinecone (namespace) and Weaviate (tenant) honor it; Qdrant, Chroma and
    # FAISS raise ConfigError when it is set rather than mix tenants.
    # An empty string is the shared default partition on both: Pinecone's
    # default namespace is "", and Weaviate treats "" as unset (no tenant),
    # so "" gives no isolation on either.
    namespace: str | None = None
