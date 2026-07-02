from uuid import UUID
 
from agent_platform.models.chunk import Chunk, ChunkMetadata
 
 
def from_tesseract(
    data: dict,
    document_id: UUID,
    min_confidence: float = 0.0,
) -> list[Chunk]:
 
    chunks: list[Chunk] = []
    index = 0
 
    for i, raw_text in enumerate(data["text"]):
        text = raw_text.strip()
        if not text:
            continue
 
        raw_conf = data["conf"][i]
        confidence = float(raw_conf) / 100 if float(raw_conf) >= 0 else 0.0
 
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
                        "page": data["page_num"][i],
                        "bbox": {
                            "left": data["left"][i],
                            "top": data["top"][i],
                            "width": data["width"][i],
                            "height": data["height"][i],
                        },
                    },
                ),
            )
        )
        index += 1
 
    return chunks