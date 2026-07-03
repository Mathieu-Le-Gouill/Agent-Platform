from __future__ import annotations

import torch

from agent_platform.models.enums.dtype import DataType


TORCH_TO_AUDIO_SAMPLE_TYPE: dict[torch.dtype, DataType] = {
    torch.int16: DataType.INT16,
    torch.int8: DataType.INT8,
    torch.uint8: DataType.UINT8,
    torch.float32: DataType.FLOAT32,
    torch.float64: DataType.FLOAT32,  # usually normalize to float32 in practice
}


AUDIO_SAMPLE_TYPE_TO_TORCH: dict[DataType, torch.dtype] = {
    DataType.INT16: torch.int16,
    DataType.INT8: torch.int8,
    DataType.UINT8: torch.uint8,
    DataType.FLOAT32: torch.float32,
}


def from_torch(dtype: torch.dtype) -> DataType:
    try:
        return TORCH_TO_AUDIO_SAMPLE_TYPE[dtype]
    except KeyError:
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    

def to_torch(sample_type: DataType) -> torch.dtype:
    try:
        return AUDIO_SAMPLE_TYPE_TO_TORCH[sample_type]
    except KeyError:
        raise ValueError(f"Unsupported audio sample type: {sample_type}")