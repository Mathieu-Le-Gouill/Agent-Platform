from __future__ import annotations

import base64
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch import Tensor

from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.document import AudioDocument
from agent_platform.core.schemas.enums import AudioFormat, DataType


def _to_numpy_dtype(sample_type: DataType) -> type[np.generic] | None:
    import numpy as np

    match sample_type:
        case DataType.INT16:
            return np.int16
        case DataType.FLOAT32:
            return np.float32
        case DataType.INT8:
            return np.int8
        case DataType.UINT8:
            return np.uint8


def _from_numpy_dtype(dtype: np.dtype) -> DataType:
    import numpy as np

    match dtype:
        case np.int16:
            return DataType.INT16
        case np.int8:
            return DataType.INT8
        case np.uint8:
            return DataType.UINT8
        case np.float32:
            return DataType.FLOAT32
        case np.float64:
            # no dedicated FLOAT64 variant; float64 arrays are represented as FLOAT32
            return DataType.FLOAT32
        case _:
            raise ValueError(f"Unsupported numpy dtype: {dtype}")


class AudioIO:
    @staticmethod
    def from_file(file_path: str) -> AudioDocument:
        info = sf.info(file_path)
        data, sample_rate = sf.read(file_path, dtype="float32", always_2d=True)

        return AudioDocument(
            source=file_path,
            content=data.tobytes(),
            format=AudioFormat(info.format) if info.format else AudioFormat.UNKNOWN,
            sample_rate=sample_rate,
            channels=info.channels,
            duration=info.frames / sample_rate if sample_rate else None,
            subtype=info.subtype,
        )

    @staticmethod
    def from_tensor(tensor: Tensor, sample_rate: int, **kwargs: Any) -> AudioChunk:
        tensor = tensor.to(torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        channels = tensor.shape[0]
        arr = tensor.detach().cpu().contiguous().numpy()

        return AudioChunk(
            data=arr.tobytes(),
            sample_rate=sample_rate,
            channels=channels,
            dtype=_from_numpy_dtype(arr.dtype),
            **kwargs,
        )

    @staticmethod
    def from_bytes(raw: bytes, sample_rate: int, **kwargs: Any) -> AudioChunk:
        return AudioChunk(data=raw, sample_rate=sample_rate, **kwargs)

    @staticmethod
    def from_base64(encoded: str, sample_rate: int, **kwargs: Any) -> AudioChunk:
        return AudioIO.from_bytes(base64.b64decode(encoded), sample_rate, **kwargs)

    @staticmethod
    def to_tensor(segment: AudioChunk) -> Tensor:
        np_dtype = _to_numpy_dtype(segment.dtype)
        arr: np.ndarray = np.frombuffer(segment.data, dtype=np_dtype)
        if segment.channels > 1:
            arr = arr.reshape(segment.channels, -1)
        # copy: np.frombuffer is a read-only view, and torch.from_numpy requires writable data
        return torch.from_numpy(arr.copy())

    @staticmethod
    def to_numpy(segment: AudioChunk) -> np.ndarray:
        np_dtype = _to_numpy_dtype(segment.dtype)
        arr: np.ndarray = np.frombuffer(segment.data, dtype=np_dtype)
        if segment.channels > 1:
            arr = arr.reshape(segment.channels, -1)
        return arr.copy()

    @staticmethod
    def to_base64(segment: AudioChunk) -> str:
        return base64.b64encode(segment.data).decode("utf-8")
