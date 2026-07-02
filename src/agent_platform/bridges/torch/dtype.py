from __future__ import annotations

import torch

from agent_platform.models.enums.dtype import AudioSampleType


TORCH_TO_AUDIO_SAMPLE_TYPE: dict[torch.dtype, AudioSampleType] = {
    torch.int16: AudioSampleType.INT16,
    torch.int8: AudioSampleType.INT8,
    torch.uint8: AudioSampleType.UINT8,
    torch.float32: AudioSampleType.FLOAT32,
    torch.float64: AudioSampleType.FLOAT32,  # usually normalize to float32 in practice
}


AUDIO_SAMPLE_TYPE_TO_TORCH: dict[AudioSampleType, torch.dtype] = {
    AudioSampleType.INT16: torch.int16,
    AudioSampleType.INT8: torch.int8,
    AudioSampleType.UINT8: torch.uint8,
    AudioSampleType.FLOAT32: torch.float32,
}


def from_torch(dtype: torch.dtype) -> AudioSampleType:
    try:
        return TORCH_TO_AUDIO_SAMPLE_TYPE[dtype]
    except KeyError:
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    

def to_torch(sample_type: AudioSampleType) -> torch.dtype:
    try:
        return AUDIO_SAMPLE_TYPE_TO_TORCH[sample_type]
    except KeyError:
        raise ValueError(f"Unsupported audio sample type: {sample_type}")