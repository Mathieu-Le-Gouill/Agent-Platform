from sentence_transformers import SentenceTransformer
from agent_platform.models.embeddable import Embeddable
from agent_platform.models.embedding import Embedding
from agent_platform.models.score import Score
from adapters.embeddings.bridges._torch_bridge import from_tensor, to_tensor

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
        item: Embeddable,
    ) -> Embedding:
        
        tensor = self.client.encode(item.content, convert_to_tensor=True)

        return from_tensor(tensor, model=self.model)


    async def similarity(
        self,
        embeddings1: Embedding,
        embeddings2: Embedding,
    ) -> Score:
        
        tensor = self.client.similarity(to_tensor(embeddings1), to_tensor(embeddings2))

        return Score.similarity(tensor.item(), low=-1.0, high=1.0)
        