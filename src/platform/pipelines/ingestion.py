"""def chunk_document(doc: Document) -> list[Chunk]:
    ...
    return [
        Chunk(
            document_id=doc.id,
            text=piece,
            index=i,
            metadata=ChunkMetadata(
                source=doc.source,
                language=doc.language,
                title=doc.title,
                ...
            )
        )
        for i, piece in enumerate(pieces)
    ]"""