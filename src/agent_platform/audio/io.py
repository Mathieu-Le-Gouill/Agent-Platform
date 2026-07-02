from __future__ import annotations

import base64
import numpy as np
import soundfile as sf
import torch
from torch import Tensor

from agent_platform.models.document import AudioDocument, DocumentMetadata, AudioInfo, AudioProperties
from agent_platform.models.chunk import AudioChunk
from agent_platform.models.enums.file_format import AudioFormat
from agent_platform.utils.uuid import new_uuid
from agent_platform.bridges.numpy.dtype import from_numpy_dtype, to_numpy_dtype


class AudioIO:

    # --- Constructors ---

    @staticmethod
    def from_file(file_path: str) -> AudioDocument:
        info = sf.info(file_path)
        data, sample_rate = sf.read(file_path, dtype="float32", always_2d=True)

        channels = info.channels
        duration = info.frames / sample_rate

        return AudioDocument(
            id=new_uuid(),
            source=file_path,
            content=data.tobytes(),
            info=AudioInfo(
                format=AudioFormat(info.format),
                language=None,
            ),
            properties=AudioProperties(
                sample_rate=sample_rate,
                channels=channels,
                duration=duration,
                bitrate=None,
                subtype=info.subtype,
            ),
            metadata=DocumentMetadata()
        )
    

    @staticmethod
    def from_tensor(tensor: Tensor, sample_rate: int, **kwargs) -> AudioChunk:
        arr = tensor.detach().cpu().contiguous().numpy()

        tensor = tensor.to(torch.float32)

        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        channels = tensor.shape[0]

        arr = tensor.numpy()

        return AudioChunk(
            id=kwargs.get("id", new_uuid()),
            data=arr.tobytes(),
            sample_rate=sample_rate,
            channels=channels,
            dtype=from_numpy_dtype(arr.dtype),
            **kwargs
        )
    

    @staticmethod
    def from_bytes(raw: bytes, sample_rate: int, **kwargs) -> AudioChunk:
        return AudioChunk(
            id=kwargs.get("id", new_uuid()),
            data=raw,
            sample_rate=sample_rate,
            **kwargs
        )


    @staticmethod
    def from_base64(encoded: str, sample_rate: int, **kwargs) -> AudioChunk:
        return AudioIO.from_bytes(
            base64.b64decode(encoded), sample_rate, **kwargs
        )

    # --- Conversions ---

    @staticmethod
    def to_tensor(segment: AudioChunk) -> Tensor:
        np_dtype = to_numpy_dtype(segment.dtype)

        arr = np.frombuffer(segment.data, dtype=np_dtype)

        if segment.channels > 1:
            arr = arr.reshape(segment.channels, -1)

        return torch.from_numpy(arr.copy())


    @staticmethod
    def to_numpy(segment: AudioChunk) -> np.ndarray:
        np_dtype = to_numpy_dtype(segment.dtype)

        arr = np.frombuffer(segment.data, dtype=np_dtype)

        if segment.channels > 1:
            arr = arr.reshape(segment.channels, -1)

        return arr.copy()


    @staticmethod
    def to_base64(segment: AudioChunk) -> str:
        return base64.b64encode(segment.data).decode("utf-8")