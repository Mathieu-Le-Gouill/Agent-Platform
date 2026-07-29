from __future__ import annotations

from uuid import UUID, uuid4

from agent_platform.core.schemas.bounding_box import BoundingBox
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import Score, ScoreKind

__all__ = ["from_tesseract"]


def from_tesseract(
    data: dict, document_id: UUID, min_confidence: float
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    n = len(data.get("text", []))

    for i in range(n):
        text = (data.get("text") or [])[i] or ""
        conf = float((data.get("conf") or [0])[i] or 0)

        if not text.strip():
            continue
        if conf < min_confidence:
            continue

        chunks.append(
            TextChunk(
                id=uuid4(),
                document_id=document_id,
                text=text.strip(),
                confidence=Score(
                    value=conf, kind=ScoreKind.CONFIDENCE, low=0, high=100
                ),
                bbox=BoundingBox(
                    x=float((data.get("left") or [0])[i]),
                    y=float((data.get("top") or [0])[i]),
                    width=float((data.get("width") or [0])[i]),
                    height=float((data.get("height") or [0])[i]),
                    normalized=False,
                ),
                metadata={
                    "page": (data.get("page_num") or [None] * n)[i],
                    "block_num": (data.get("block_num") or [None] * n)[i],
                },
            )
        )

    return chunks
