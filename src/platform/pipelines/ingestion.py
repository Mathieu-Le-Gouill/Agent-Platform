from langchain_text_splitters import RecursiveCharacterTextSplitter
from models.document import Document
from models.chunk import Chunk, ChunkMetadata


_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ".", " ", ""],
)


def chunk_document(doc: Document) -> list[Chunk]:
    if not doc.text:
        return []

    pieces = _splitter.split_text(doc.text)

    return [
        Chunk(
            document_id=doc.id,
            text=piece,
            index=i,
            start_char=None,    # splitter doesn't expose offsets directly
            metadata=ChunkMetadata(
                source=doc.source,
                language=doc.language,
                title=doc.title,
                document_type=doc.document_type,
            ),
        )
        for i, piece in enumerate(pieces)
    ]


# TODO langchain_experimental.text_splitter.SemanticChunker