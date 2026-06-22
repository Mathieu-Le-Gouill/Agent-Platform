
from dataclasses import dataclass, field
    

@dataclass(slots=True)
class ChunkMetadata:
    document_id: str
    chunk_index: int
    start_char: int
    end_char: int
    extra: dict[str, str | int | float | bool] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    id: str
    content: str
    metadata: ChunkMetadata | None