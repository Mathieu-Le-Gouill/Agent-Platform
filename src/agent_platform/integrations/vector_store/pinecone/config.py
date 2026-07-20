from __future__ import annotations
from agent_platform.core.interfaces.vector_store.config import VectorStoreConfig


class PineconeConfig(VectorStoreConfig):
    # Skips a describe-index round trip when set (index host, not the deprecated
    # pod "environment" concept).
    host: str | None = None

    # Note: this provider only queries/upserts against a pre-existing index; it
    # never calls create_index, so ServerlessSpec's `cloud`/`region` fields are
    # intentionally not modeled here. Add them if index-admin support is needed.


# sources: https://docs.pinecone.io/guides/index-data/create-an-index
