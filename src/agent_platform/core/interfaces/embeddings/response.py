from dataclasses import dataclass
from agent_platform.core.schemas.embedding import Embedding


@dataclass(slots=True, frozen=True)
class EmbeddingResponse:
    embeddings: list[Embedding]
    model: str

    def __iter__(self):
        return iter(self.embeddings)

    def __len__(self):
        return len(self.embeddings)
