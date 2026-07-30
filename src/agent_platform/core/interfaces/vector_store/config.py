from agent_platform.core.config import RequestOptions


class VectorStoreConfig(RequestOptions):
    collection_name: str = (
        "default"  # index/collection identifier to read and write vectors from
    )
    namespace: str | None = (
        None  # optional logical partition within a collection for multi-tenant isolation
    )
