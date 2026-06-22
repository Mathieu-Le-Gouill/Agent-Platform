import numpy as np
import soundfile as sf
from torch import Tensor
import torch
import base64
from dataclasses import dataclass


@dataclass(slots=True)
class AudioSegment():
    data: Tensor
    sample_rate: int
    metadata: dict | None
    
    # --- Constructors ---

    def __init__(
        self,
        data: Tensor,
        sample_rate: int,
        metadata: dict | None = None,
    ) -> None:
        
        self.data = data
        self.sample_rate = sample_rate
        self.metadata = metadata

    @classmethod
    def from_file(
        cls,
        file_path: str,
    ) -> "AudioSegment": 
        
        data, sample_rate = sf.read(file_path)
        tensor = torch.from_numpy(data)
        
        return cls(
            data=tensor,
            sample_rate=sample_rate,
        )

    @classmethod
    def from_bytes(
        cls,
        bytes: bytes,
    ) -> "AudioSegment":
        ...


    @classmethod
    def from_tensor(
        cls,
        tensor: Tensor,
    ) -> "AudioSegment":
        ...


    @classmethod
    def from_url(
        cls,
        url: str,
    ) -> "AudioSegment":
        ...


    # --- Attributes ---

    @property
    def channels(self) -> int:
        return self.data.shape[1] if self.data.ndim > 1 else 1

    @property
    def duration(self) -> float:
        return len(self.data) / self.sample_rate

    
    # --- Audio Tools ---

    def resample(
        self,
        sample_rate: int,
    ) -> None :
        ...

    
    def chunck(
        self,
        seconds: int,
    ) -> np.ndarray:
        ...


    # --- Conversion Tools ---

    def to_torch(self) -> Tensor:
        return self.data
    

    def to_numpy(self) -> np.ndarray:
        return self.data.cpu().numpy()
    
    
    def to_bytes(self) -> bytes:
        return self.data.cpu().numpy().tobytes()
    
    
    def to_base64(self) -> str:
        return base64.b64encode(self.to_bytes()).decode("utf-8")

    
