from __future__ import annotations

import base64
import numpy as np
import soundfile as sf
import torch
from torch import Tensor

from models.audio import AudioSegment


class AudioSegmentIO:

    # --- Constructors ---

    @staticmethod
    def from_file(file_path: str) -> AudioSegment:
        data, sample_rate = sf.read(file_path, dtype="float32", always_2d=False)
        channels = data.shape[1] if data.ndim > 1 else 1
        return AudioSegment(
            data=data.tobytes(),
            sample_rate=sample_rate,
            channels=channels,
        )
    

    @staticmethod
    def from_tensor(tensor: Tensor, sample_rate: int) -> AudioSegment:
        arr = tensor.cpu().numpy()
        channels = arr.shape[0] if arr.ndim > 1 else 1
        return AudioSegment(
            data=arr.tobytes(),
            sample_rate=sample_rate,
            channels=channels,
        )
    

    @staticmethod
    def from_bytes(raw: bytes, sample_rate: int, channels: int = 1) -> AudioSegment:
        return AudioSegment(data=raw, sample_rate=sample_rate, channels=channels)


    @staticmethod
    def from_base64(encoded: str, sample_rate: int, channels: int = 1) -> AudioSegment:
        return AudioSegmentIO.from_bytes(
            base64.b64decode(encoded), sample_rate, channels
        )

    # --- Conversions ---

    @staticmethod
    def to_tensor(segment: AudioSegment) -> Tensor:
        arr = np.frombuffer(segment.data, dtype=np.float32)
        if segment.channels > 1:
            arr = arr.reshape(-1, segment.channels)
        return torch.from_numpy(arr.copy())


    @staticmethod
    def to_numpy(segment: AudioSegment) -> np.ndarray:
        return np.frombuffer(segment.data, dtype=np.float32).copy()


    @staticmethod
    def to_base64(segment: AudioSegment) -> str:
        return base64.b64encode(segment.data).decode("utf-8")