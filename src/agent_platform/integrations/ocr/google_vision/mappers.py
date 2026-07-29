from __future__ import annotations

from uuid import UUID, uuid4

from google.cloud import vision

from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import Score

__all__ = ["from_google_vision"]


def from_google_vision(
    response: vision.AnnotateImageResponse, document_id: UUID, min_confidence: float
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    pages = response.full_text_annotation.pages

    for page_num, page in enumerate(pages):
        for block in page.blocks:
            for paragraph in block.paragraphs:
                words = paragraph.words
                if not words:
                    continue

                text = " ".join(
                    "".join(symbol.text for symbol in word.symbols) for word in words
                )
                confs = [
                    symbol.confidence
                    for word in words
                    for symbol in word.symbols
                    if symbol.confidence
                ]
                avg_conf = (sum(confs) / len(confs)) if confs else 0.0

                if avg_conf < min_confidence:
                    continue

                chunks.append(
                    TextChunk(
                        id=uuid4(),
                        document_id=document_id,
                        text=text.strip(),
                        confidence=Score.confidence(avg_conf),
                        metadata={
                            "page": page_num,
                        },
                    )
                )

    return chunks
