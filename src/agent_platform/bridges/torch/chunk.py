import io
import torch
import torchaudio
from typing import Sequence

from agent_platform.models.chunk import AudioChunk

def audio_segment_to_torch(audio_sequence: Sequence[AudioChunk]) -> torch.Tensor:
    """
    Convert a sequence of AudioChunk objects to a torch tensor.
    Each chunk.data is expected to be audio file bytes (wav/flac/mp3).
    """

    audio_tensors = []

    for chunk in audio_sequence:
        buffer = io.BytesIO(chunk.data)

        waveform, sample_rate = torchaudio.load(buffer)

        # optional: ensure sample rate consistency
        if sample_rate != chunk.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=chunk.sample_rate,
            )

        audio_tensors.append(waveform)

    # Shape: (num_chunks, channels, time)
    return torch.stack(audio_tensors)