from langchain_text_splitters import RecursiveCharacterTextSplitter
from agent_platform.models.document import TextDocument
from agent_platform.models.chunk import TextChunk


_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ".", " ", ""],
)


def chunk_document(doc: TextDocument) -> list[TextChunk]:
    text = doc.text if hasattr(doc, "text") and isinstance(doc.text, str) else ""
    if not text:
        return []

    pieces = _splitter.split_text(text)

    return [
        TextChunk(
            document_id=doc.id,
            text=piece,
            index=i,
            metadata={
                "source": doc.source,
                "language": doc.language.value if hasattr(doc, "language") and doc.language else None,
                "title": doc.metadata.title,
            },
        )
        for i, piece in enumerate(pieces)
    ]
