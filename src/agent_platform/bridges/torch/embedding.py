import torch
from agent_platform.models.embedding import Embedding
from uuid import UUID


def to_torch(embedding: Embedding, device: str = "cpu") -> torch.Tensor:
    return torch.tensor(
        embedding.vector,
        dtype=torch.float32,
        device=device,
    )


def from_torch(
    tensor: torch.Tensor,
    *,
    model: str,
    ids: list[UUID],
) -> list[Embedding]:
    
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)

    if tensor.shape[0] != len(ids):
        raise ValueError(
            f"Number of embeddings ({tensor.shape[0]}) "
            f"does not match number of ids ({len(ids)})"
        )

    return [
        Embedding.from_list(
            vector.tolist(),
            model=model,
            id=id,
        )
        for vector, id in zip(tensor, ids)
    ]