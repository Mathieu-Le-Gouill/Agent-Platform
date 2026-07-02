from abc import ABC, abstractmethod
from typing import Sequence

from agent_platform.integrations.embeddings.response import EmbeddingResponse
from agent_platform.models.protocols.text_unit import TextUnit


class BaseEmbeddingProvider(ABC):

    @abstractmethod
    async def encode(
        self,
        items: Sequence[TextUnit],
    ) -> EmbeddingResponse:
        ...