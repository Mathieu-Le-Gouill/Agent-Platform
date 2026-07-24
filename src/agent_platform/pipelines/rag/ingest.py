from agent_platform.core.interfaces.vector_store.port import VectorStore
from agent_platform.core.schemas.chunk import TextChunk


async def ingest(chunks: list[TextChunk], store: VectorStore) -> None:
    await store.add(chunks)
