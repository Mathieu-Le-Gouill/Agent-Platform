from integrations.vector_store.port import VectorStore
from models.document import Document


async def ingest(docs: list[Document], store: VectorStore) -> None:
    await store.add(docs)