from uuid import UUID
 
from agent_platform.models.chunk import Chunk, ChunkMetadata
 
 
def from_google_vision(
    response,
    document_id: UUID,
    min_confidence: float = 0.0,
) -> list[Chunk]:
 
    chunks: list[Chunk] = []
    index = 0
 
    for page_num, page in enumerate(response.full_text_annotation.pages, start=1):
        for block in page.blocks:
            for paragraph in block.paragraphs:
                words = [
                    "".join(symbol.text for symbol in word.symbols)
                    for word in paragraph.words
                ]
                confidences = [word.confidence for word in paragraph.words]
 
                text = " ".join(words).strip()
                if not text:
                    continue
 
                confidence = sum(confidences) / len(confidences) if confidences else 0.0
 
                if confidence < min_confidence:
                    continue
 
                chunks.append(
                    Chunk(
                        document_id=document_id,
                        text=text,
                        index=index,
                        metadata=ChunkMetadata(
                            extra={
                                "confidence": confidence,
                                "page": page_num,
                            },
                        ),
                    )
                )
                index += 1
 
    return chunks