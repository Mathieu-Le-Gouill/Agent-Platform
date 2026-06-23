from sentence_transformers import SentenceTransformer
from torch import Tensor

class SentenceTransformerEmbedder:
    model: SentenceTransformer


    def __init__(
        self,
        model: str,
    ) -> None:
        self.model = SentenceTransformer(model)


    def encode(
        self,
        data: Tensor,
    ) -> Tensor:
        return self.model.encode(data)


    def similaity(
        self,
        embeddings1,
        embeddings2,
    ) -> Tensor:
        return self.model.similarity(embeddings1, embeddings2)
        