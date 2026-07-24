from __future__ import annotations

from agent_platform.utils.lazy_imports import make_lazy_provider_accessors

_PROVIDERS: dict[str, str] = {
    "ChromaStore": "agent_platform.integrations.vector_store.chroma.chroma",
    "FAISSStore": "agent_platform.integrations.vector_store.faiss.faiss",
    "PineconeStore": "agent_platform.integrations.vector_store.pinecone.pinecone",
    "QdrantVectorStoreProvider": "agent_platform.integrations.vector_store.qdrant.qdrant",
    "WeaviateStore": "agent_platform.integrations.vector_store.weaviate.weaviate",
}

__getattr__, __dir__ = make_lazy_provider_accessors(_PROVIDERS, globals())
