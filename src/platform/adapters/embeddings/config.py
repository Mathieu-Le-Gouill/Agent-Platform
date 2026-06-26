from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class EmbeddingConfig:
    batch_size: int = 32
    normalize: bool = True          # L2 normalize - needed for cosine similarity
    truncate: bool = True           # silently truncate if text exceeds model max length
    convert_to_tensor: bool = True
    dimensions: int | None = None   # Matryoshka truncation (OpenAI text-embedding-3+)
    device: str | None = None
    show_progress: bool = False
    