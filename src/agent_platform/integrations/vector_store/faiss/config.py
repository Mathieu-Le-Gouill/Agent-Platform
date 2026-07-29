from __future__ import annotations

from enum import StrEnum

from agent_platform.core.interfaces.vector_store.config import VectorStoreConfig


class DistanceMetric(StrEnum):
    COSINE = "cosine"  # angular similarity, scale-invariant; most common default for text embeddings
    DOT = "dot"  # raw dot product; favors vector magnitude, used when embeddings are pre-normalized
    EUCLIDEAN = "euclidean"  # straight-line (L2) distance; smaller is more similar


class FAISSConfig(VectorStoreConfig):
    # Local on-disk index directory: holds `index.faiss` (the native FAISS
    # index) and `docstore.pkl` (chunk metadata + id mapping), written by
    # `FAISSStore.add()`/read by `FAISSStore._load_or_none()`.
    index_path: str | None = None

    # Selects the native FAISS index type/similarity transform directly in
    # faiss/provider.py (`_new_index`/`_to_similarity`). Only FAISS calls
    # create_index/create_collection through this platform; other vector
    # stores connect to a pre-existing collection whose metric was already
    # chosen at index-creation time, so this field has no equivalent there.
    distance: DistanceMetric = DistanceMetric.COSINE


# sources: https://github.com/facebookresearch/faiss/wiki
