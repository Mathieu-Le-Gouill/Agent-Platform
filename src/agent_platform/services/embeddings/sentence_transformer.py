from sentence_transformers import SentenceTransformer
from models.protocols.text_unit import TextUnit
from models.embedding import Embedding
from models.score import Score
from bridges.embedding.torch import from_tensor, to_tensor

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
        item: TextUnit,
    ) -> Embedding:
        
        tensor = self.client.encode(item.text, convert_to_tensor=True)

        return from_tensor(tensor, model=self.model)


    async def similarity(
        self,
        embeddings1: Embedding,
        embeddings2: Embedding,
    ) -> Score:
        
        tensor = self.client.similarity(to_tensor(embeddings1), to_tensor(embeddings2))

        return Score.similarity(tensor.item(), low=-1.0, high=1.0)
        