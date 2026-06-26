"""
async def ingest(doc: Document) -> None:
    chunks = chunk_document(doc)
    embeddings = await embedder.embed_batch(chunks)
    await store.upsert(chunks, embeddings)
"""