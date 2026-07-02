from agent_platform.integrations.vector_store.port import VectorStore
from agent_platform.models.document import Document


async def ingest(docs: list[Document], store: VectorStore) -> None:
    await store.add(docs)