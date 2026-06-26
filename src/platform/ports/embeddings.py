from typing import Protocol

from adapters.embeddings.response import EmbeddingResponse
from adapters.embeddings.config import EmbeddingConfig
from models.protocols.text_unit import TextUnit


class EmbeddingProvider(Protocol):

    async def encode(
        self,
        items: list[TextUnit],
        config: EmbeddingConfig = EmbeddingConfig(),
    ) -> EmbeddingResponse: 
        ...