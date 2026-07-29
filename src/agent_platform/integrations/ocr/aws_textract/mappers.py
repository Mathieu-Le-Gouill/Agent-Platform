from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from agent_platform.core.schemas.bounding_box import BoundingBox
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import Score, ScoreKind

__all__ = ["from_textract"]


def from_textract(
    response: dict[str, Any], document_id: UUID, min_confidence: float
) -> list[TextChunk]:
    chunks: list[TextChunk] = []

    for block in response.get("Blocks", []):
        if block.get("BlockType") != "LINE":
            continue

        conf = block.get("Confidence", 0) or 0
        if conf < min_confidence:
            continue

        text = block.get("Text", "") or ""
        bbox = block.get("Geometry", {}).get("BoundingBox", {})

        chunks.append(
            TextChunk(
                id=uuid4(),
                document_id=document_id,
                text=text.strip(),
                confidence=Score(
                    value=conf, kind=ScoreKind.CONFIDENCE, low=0, high=100
                ),
                bbox=BoundingBox(
                    x=bbox.get("Left", 0.0),
                    y=bbox.get("Top", 0.0),
                    width=bbox.get("Width", 0.0),
                    height=bbox.get("Height", 0.0),
                    normalized=True,
                ),
                metadata={
                    "page": block.get("Page"),
                },
            )
        )

    return chunks
