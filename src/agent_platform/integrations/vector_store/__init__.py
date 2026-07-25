from __future__ import annotations

from agent_platform.utils.lazy_imports import make_lazy_provider_accessors

_PROVIDERS: dict[str, str] = {
    "ChromaStore": "agent_platform.integrations.vector_store.chroma.provider",
    "FAISSStore": "agent_platform.integrations.vector_store.faiss.provider",
    "PineconeStore": "agent_platform.integrations.vector_store.pinecone.provider",
    "QdrantVectorStoreProvider": "agent_platform.integrations.vector_store.qdrant.provider",
    "WeaviateStore": "agent_platform.integrations.vector_store.weaviate.provider",
}

__getattr__, __dir__ = make_lazy_provider_accessors(_PROVIDERS, globals())
