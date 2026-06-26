from models.document import Document
from agent_platform.bridges.chroma.chunk import to_chroma_many


def to_chroma(document: Document) -> dict:
    return to_chroma_many(document.chunks)