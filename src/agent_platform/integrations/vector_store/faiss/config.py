from __future__ import annotations

from agent_platform.core.interfaces.vector_store.config import VectorStoreConfig


class FAISSConfig(VectorStoreConfig):
    # Local on-disk index directory: holds `index.faiss` (the native FAISS
    # index) and `docstore.pkl` (chunk metadata + id mapping), written by
    # `FAISSStore.add()`/read by `FAISSStore._load_or_none()`.
    index_path: str | None = None

    # Note: base `distance` selects the native index type/similarity
    # transform directly in faiss/provider.py (`_new_index`/`_to_similarity`),
    # not a LangChain `distance_strategy`.


# sources: https://github.com/facebookresearch/faiss/wiki
