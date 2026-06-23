from typing import Protocol
from torch import Tensor

class EmbedderPort(Protocol):
    async def encode(
        self,
        data: Tensor,
    ) -> str: 
        ...