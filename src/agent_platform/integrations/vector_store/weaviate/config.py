from __future__ import annotations
from agent_platform.core.interfaces.vector_store.config import VectorStoreConfig


class WeaviateConfig(VectorStoreConfig):
    text_key: str = "text"  # property name storing the document body in each Weaviate object

    # gRPC port for `weaviate.connect_to_custom`/`connect_to_local`; Weaviate
    # Cloud and most self-hosted setups keep the 50051 default.
    grpc_port: int = 50051

    # When set, used directly instead of parsing host/port out of the
    # credentials URL string (fragile on IPv6/paths/missing scheme).
    http_host: str | None = None

    # Paired with `http_host`; falls back to the URL-parsed port when unset.
    http_port: int | None = None


"""
sources: https://python.langchain.com (WeaviateVectorStore)
         https://weaviate.io/developers/weaviate/client-libraries/python#connection
"""
