from agent_platform.core.config import ProviderConfig


class VectorStoreConfig(ProviderConfig):
    collection_name: str = (
        "default"  # index/collection identifier to read and write vectors from
    )
    namespace: str | None = (
        None  # optional logical partition within a collection for multi-tenant isolation
    )
