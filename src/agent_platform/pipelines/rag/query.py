from typing import Any

from agent_platform.core.interfaces.vector_store.config import VectorStoreConfig
from agent_platform.core.interfaces.vector_store.port import VectorStore
from agent_platform.core.schemas.chunk import TextChunk


async def query(
    vector: list[float],
    store: VectorStore,
    k: int = 5,
    config: VectorStoreConfig | None = None,
    filter: dict[str, Any] | None = None,
) -> list[TextChunk]:
    return await store.search(vector, k=k, config=config, filter=filter)
