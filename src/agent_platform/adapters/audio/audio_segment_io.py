from __future__ import annotations
import base64
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch import Tensor
from core.value_objects.audio_segment import AudioSegment


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

    # --- Tools ---

    @staticmethod
    def resample(segment: AudioSegment, target_rate: int) -> AudioSegment:
        if segment.sample_rate == target_rate:
            return segment
        tensor = AudioSegmentIO.to_tensor(segment).unsqueeze(0)
        resampled = torchaudio.functional.resample(
            tensor, segment.sample_rate, target_rate
        ).squeeze(0)
        return AudioSegmentIO.from_tensor(resampled, target_rate)


    @staticmethod
    def chunk(segment: AudioSegment, seconds: float) -> list[AudioSegment]:
        arr = AudioSegmentIO.to_numpy(segment)
        size = int(seconds * segment.sample_rate)
        return [
            AudioSegment(
                data=arr[i : i + size].tobytes(),
                sample_rate=segment.sample_rate,
                channels=segment.channels,
                start_ms=int(i / segment.sample_rate * 1000),
                end_ms=int(min(i + size, len(arr)) / segment.sample_rate * 1000),
            )
            for i in range(0, len(arr), size)
        ]