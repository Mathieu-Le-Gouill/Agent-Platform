from typing import Protocol, Sequence

from integrations.embeddings.response import EmbeddingResponse
from models.protocols.text_unit import TextUnit


class EmbeddingProvider(Protocol):

    async def encode(
        self,
        items: Sequence[TextUnit],
    ) -> EmbeddingResponse: 
        ...