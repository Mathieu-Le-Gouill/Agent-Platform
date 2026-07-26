from __future__ import annotations

from agent_platform.core.interfaces.vector_store.config import VectorStoreConfig


class FAISSConfig(VectorStoreConfig):
    # Local on-disk index path for FAISS.save_local/load_local.
    index_path: str | None = None

    # Note: `distance` (base field) is mapped to LangChain's `distance_strategy`
    # in faiss.py — see DISTANCE_STRATEGY_MAP there.


# sources: https://reference.langchain.com/python/langchain-community/vectorstores/faiss/FAISS
