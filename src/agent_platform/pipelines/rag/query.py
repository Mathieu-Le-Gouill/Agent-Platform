from agent_platform.integrations.vector_store.port import VectorStore
from agent_platform.models.chunk import TextChunk


async def query(vector: list[float], store: VectorStore, k: int = 5) -> list[TextChunk]:
    return await store.search(vector, k=k)
