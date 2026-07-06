from agent_platform.integrations.vector_store.port import VectorStore
from agent_platform.models.chunk import TextChunk


async def ingest(chunks: list[TextChunk], store: VectorStore) -> None:
    await store.add(chunks)
