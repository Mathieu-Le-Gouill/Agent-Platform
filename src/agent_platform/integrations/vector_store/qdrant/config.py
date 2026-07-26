from __future__ import annotations

from agent_platform.core.interfaces.vector_store.config import VectorStoreConfig


class QdrantConfig(VectorStoreConfig):
    url: str = "http://localhost:6333"  # Qdrant REST/gRPC endpoint, passed to QdrantClient(url=...)

    # Lower-latency transport for collection ops; forwarded to QdrantClient.
    prefer_grpc: bool = False

    # Note: `distance` (base field) is intentionally unused here — this provider
    # assumes the collection already exists and never calls create_collection.
    # If that changes, Qdrant's Distance enum spells differently than
    # DistanceMetric (`Distance.EUCLID`, not `EUCLIDEAN`) and needs a mapping table.

    # `add_hybrid`/`search_hybrid` additionally assume the collection was
    # provisioned with two named vectors: a dense one under `"dense"` and a
    # sparse one under `"sparse"` configured with `models.Modifier.IDF`, so
    # Qdrant computes IDF weighting server-side from the raw term-frequency
    # values this provider sends (see `integrations/embeddings/bm25`).


"""
sources: https://qdrant.tech/documentation/interfaces
         https://github.com/qdrant/qdrant-client
"""
