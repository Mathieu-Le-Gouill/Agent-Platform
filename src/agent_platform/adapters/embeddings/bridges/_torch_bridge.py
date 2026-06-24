import torch
from core.entities.embedding import Embedding


def to_tensor(embedding: Embedding, device: str = "cpu") -> torch.Tensor:
    return torch.tensor(embedding.vector, dtype=torch.float32, device=device)


def from_tensor(tensor: torch.Tensor, *, model: str) -> Embedding:
    return Embedding.from_list(tensor.squeeze().tolist(), model=model)