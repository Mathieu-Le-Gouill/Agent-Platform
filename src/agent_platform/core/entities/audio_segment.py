from enum import Enum
import numpy as np
import soundfile as sf
import torch
import base64
from io import BytesIO

class AudioSegment():

    # --- Constructors ---

    def __init__(
        self,
        data: torch.Tensor,
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


    def from_file(
        self,
        file_path: str,
    ) -> None: 
        
        data, sample_rate = sf.read(file_path)
        self.data = torch.from_numpy(data)
        self.sample_rate = sample_rate


    def from_bytes(
        self,
        bytes: bytes,
    ) -> None:
        ...


    def from_tensor(
        self,
        tensor: torch.Tensor,
    ) -> None:
        ...


    def from_url(
        self,
        url: str,
    ) -> None:
        ...

    
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

    def torch(self) -> torch.Tensor:
        return self.data
    

    def numpy(self) -> np.ndarray:
        return self.data.cpu().numpy()
    
    
    def bytes(self) -> bytes:
        return bytes(self.data) if self.data.dim == 1 else self.data.cpu().numpy().tobytes()
    
    
    def base64(self) -> str:
        buffer = BytesIO()
        torch.save(self.data, buffer)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    
