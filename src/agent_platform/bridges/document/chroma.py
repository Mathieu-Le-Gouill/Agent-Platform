# bridges/document/chroma.py
from models.document import Document
from models.chunk import Chunk
from bridges.chunk.chroma import to_chroma_many, from_chroma_many


def to_chroma(documents: list[Document]) -> dict:
    chunks = [chunk for document in documents for chunk in document.chunks]
    return to_chroma_many(chunks)


def from_chroma(results: dict) -> list[Chunk]:
    return from_chroma_many(results)