import torch
from torch import Tensor
from dataclasses import dataclass


@dataclass(slots=True)
class VideoSegement():

    # --- Constructors ---

    def __init__(
        self,
        data: Tensor,
        sample_rate: int,
        channels: int,
        duration: int,
        metadata: dict | None = None,
    ) -> None:
        
        self.data = data
        self.sample_rate = sample_rate
        self.channels = channels
        self.duration = duration
        self.metadata = metadata