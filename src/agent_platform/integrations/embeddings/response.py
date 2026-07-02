from dataclasses import dataclass
from agent_platform.models.embedding import Embedding
from agent_platform.models.token import TokenUsage


@dataclass(slots=True, frozen=True)
class EmbeddingResponse:
    embeddings: list[Embedding]
    model: str
    usage: TokenUsage

    def __iter__(self):
        return iter(self.embeddings)

    def __len__(self):
        return len(self.embeddings)