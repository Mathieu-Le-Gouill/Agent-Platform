from models.document import Document
from qdrant_client.models import PointStruct
from mappers.qdrant.chunk import to_point


def to_points(document: Document) -> list[PointStruct]:
    return [
        to_point(chunk)
        for chunk in document.chunks
    ]


