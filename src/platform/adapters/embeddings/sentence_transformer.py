from sentence_transformers import SentenceTransformer
from models.protocols.text_unit import TextUnit
from adapters.embeddings.response import EmbeddingResponse
from adapters.embeddings.config import EmbeddingConfig
from core.embedding.torch import from_torch
from models.token import TokenUsage

class SentenceTransformerEmbedder:
    client: SentenceTransformer
    model: str


    def __init__(
        self,
        model: str,
    ) -> None:
        
        self.client = SentenceTransformer(model)
        self.model = model


    async def encode(
        self,
        items: list[TextUnit],
        config: EmbeddingConfig = EmbeddingConfig(),
    ) -> EmbeddingResponse: 
        
        tensor = self.client.encode(
            [item.text for item in items],
            convert_to_tensor=True,
        )

        embeddings = from_torch(tensor, model=self.model, ids=[item.id for item in items])

        return EmbeddingResponse(embeddings=embeddings, model=self.model, usage=TokenUsage.zero())
        