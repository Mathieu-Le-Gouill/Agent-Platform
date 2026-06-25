# bridges/document/qdrant.py
from models.chunk import Chunk
from models.document import Document
from qdrant_client.models import PointStruct
from bridges.chunk.qdrant import to_point, from_point


def to_points(documents: list[Document]) -> list[PointStruct]:
    return [
        to_point(chunk)
        for document in documents
        for chunk in document.chunks
    ]


def from_points(points: list) -> list[Chunk]:
    return [from_point(point) for point in points]