from abc import ABC, abstractmethod
from typing import Sequence

from agent_platform.integrations.embeddings.response import EmbeddingResponse
from agent_platform.models.chunk import TextChunk


class BaseEmbeddingProvider(ABC):

    @abstractmethod
    async def encode(
        self,
        items: Sequence[TextChunk],
    ) -> EmbeddingResponse:
        ...