from typing import Protocol

from platform.providers.embeddings.response import EmbeddingResponse
from .config import EmbeddingConfig
from models.protocols.text_unit import TextUnit


class EmbeddingProvider(Protocol):

    async def encode(
        self,
        items: list[TextUnit],
        config: EmbeddingConfig = EmbeddingConfig(),
    ) -> EmbeddingResponse: 
        ...