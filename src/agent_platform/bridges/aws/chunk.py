from uuid import UUID
 
from agent_platform.models.chunk import Chunk, ChunkMetadata


def from_textract(
    response: dict,
    document_id: UUID,
    min_confidence: float = 0.0,
) -> list[Chunk]:
 
    chunks: list[Chunk] = []
    index = 0
 
    for block in response.get("Blocks", []):
        if block.get("BlockType") != "LINE":
            continue
 
        confidence = block.get("Confidence", 0.0) / 100
 
        if confidence < min_confidence:
            continue
 
        chunks.append(
            Chunk(
                document_id=document_id,
                text=block.get("Text", ""),
                index=index,
                metadata=ChunkMetadata(
                    extra={
                        "confidence": confidence,
                        "bbox": block.get("Geometry", {}).get("BoundingBox"),
                    },
                ),
            )
        )
        index += 1
 
    return chunks