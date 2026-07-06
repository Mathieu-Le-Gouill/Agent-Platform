from __future__ import annotations

from silero_vad import load_silero_vad, get_speech_timestamps
from typing import AsyncIterator, Sequence
from collections import deque

import torch

from agent_platform.integrations.vad.base import BaseVAD
from agent_platform.integrations.vad.configuration import SileroVadConfig
from agent_platform.integrations.vad.requirements import AudioRequirements
from agent_platform.models.chunk import AudioChunk
from agent_platform.models.span import SampleSpan
from agent_platform.models.enums import DataType
from agent_platform.audio.io import AudioIO



class SileroVAD(BaseVAD[SileroVadConfig]):

    def __init__(self):
        self.model = load_silero_vad()


    @property
    def requirements(self) -> AudioRequirements:
        return AudioRequirements(
            sample_rates=(8000, 16000),
            channels=1,
            dtype=DataType.FLOAT32,
            normalized=True,
        )
    

    def detect(
        self,
        audio_sequence: Sequence[AudioChunk],
        config: SileroVadConfig | None = None
    ) -> list[SampleSpan]: 
        
        config = config or SileroVadConfig()
        
        if not audio_sequence:
            return []
        
        chunks = []

        for chunk in audio_sequence:
            self._validate_chunk(chunk, config.sample_rate)

            chunks.append(AudioIO.to_tensor(chunk))

        torch_audio = torch.cat(chunks, dim=-1)

        speech_samples = get_speech_timestamps(
            torch_audio,
            self.model,
            sampling_rate=config.sample_rate,
            threshold=config.threshold,
            speech_pad_ms=config.speech_pad_ms,
            min_speech_duration_ms=config.min_speech_duration_ms,
            min_silence_duration_ms=config.min_silence_duration_ms,
        )

        return [
            SampleSpan(start=s["start"], end=s["end"])
            for s in speech_samples
        ]
    

    async def adetect(
        self,
        audio_sequence: AsyncIterator[AudioChunk],
        config: SileroVadConfig | None = None,
    ) -> AsyncIterator[SampleSpan]: 
        
        config = config or SileroVadConfig()

        buffer_chunks = deque()
        torch_audio = None

        async for chunk in audio_sequence:
            if chunk.sample_rate != config.sample_rate:
                raise ValueError(
                    f"Expected {config.sample_rate}, got {chunk.sample_rate}"
                )
            
            tensor = AudioIO.to_tensor(chunk)

            buffer_chunks.append(tensor)

            if torch_audio is None:
                torch_audio = tensor
            else:
                torch_audio = torch.cat((torch_audio, tensor), dim=-1)

            if torch_audio.shape[-1] > config.max_samples:
                torch_audio = torch_audio[:, -config.max_samples:]

                while buffer_chunks and sum(x.shape[-1] for x in buffer_chunks) > config.max_samples:
                    buffer_chunks.popleft()

            speech_samples = get_speech_timestamps(
                torch_audio,
                self.model,
                sampling_rate=config.sample_rate,
                threshold=config.threshold,
                speech_pad_ms=config.speech_pad_ms,
                min_speech_duration_ms=config.min_speech_duration_ms,
                min_silence_duration_ms=config.min_silence_duration_ms,
            )

            for s in speech_samples:
                yield SampleSpan(start=s["start"], end=s["end"])
    
    # Ref: https://github.com/snakers4/silero-vad