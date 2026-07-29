from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import Score

if TYPE_CHECKING:
    from mistralai.models import OCRResponse

__all__ = ["from_mistral"]


def from_mistral(
    response: OCRResponse, document_id: UUID, min_confidence: float = 0.0
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for page in response.pages:
        text = (page.markdown or "").strip()
        if not text:
            continue

        scores = getattr(page, "confidence_scores", None)
        avg_conf = (
            getattr(scores, "average_page_confidence_score", None) if scores else None
        )

        if avg_conf is not None and avg_conf < min_confidence:
            continue

        chunk_kwargs: dict[str, Any] = dict(
            id=uuid4(),
            document_id=document_id,
            text=text,
            index=page.index,
            metadata={"page": page.index},
        )
        if avg_conf is not None:
            chunk_kwargs["confidence"] = Score.confidence(avg_conf)

        chunks.append(TextChunk(**chunk_kwargs))
    return chunks
