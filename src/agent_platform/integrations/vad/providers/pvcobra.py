from __future__ import annotations

import pvcobra
from typing import AsyncIterator, Sequence
from pydantic import SecretStr

from agent_platform.integrations.vad.framebased import FrameBasedVAD
from agent_platform.integrations.vad.configuration import PvcobraVadConfig
from agent_platform.models.chunk import AudioChunk
from agent_platform.models.span import SampleSpan
from agent_platform.integrations.vad.state import VADState
from agent_platform.integrations.vad.requirements import AudioRequirements
from agent_platform.models.enums.dtype import DataType


class PvcobraVAD(FrameBasedVAD[PvcobraVadConfig]):

    def __init__(
        self, 
        acess_key: SecretStr,
        device: str | None = None, 
        library_path: str | None = None
    ) -> None:

        self.handle = pvcobra.create(acess_key.get_secret_value(), device, library_path)


    @property
    def requirements(self) -> AudioRequirements:
        return AudioRequirements(
            sample_rates=(16000,),
            channels=1,
            dtype=DataType.INT16,
            normalized=False,
        )


    def detect(
        self,
        audio_sequence: Sequence[AudioChunk],
        config: PvcobraVadConfig | None = None
    ) -> list[SampleSpan]: 
        
        config = config or PvcobraVadConfig()
        state = VADState()
        
        if not audio_sequence:
            return []
        
        voiced_frames: list[SampleSpan] = []

        for chunk in audio_sequence:
            self._validate_chunk(chunk, config.sample_rate)

            if self._is_speech(chunk, config):
                self._on_speech(state, chunk, config)
            else:
                span = self._on_silence(state, chunk, config)

                if span is not None:
                    voiced_frames.append(span)
                    
        return voiced_frames
    

    async def adetect(
        self,
        audio_sequence: AsyncIterator[AudioChunk],
        config: PvcobraVadConfig | None = None,
    ) -> AsyncIterator[SampleSpan]: 
        
        config = config or PvcobraVadConfig()
        state = VADState()

        async for chunk in audio_sequence:

            self._validate_chunk(chunk, config.sample_rate)

            if self._is_speech(chunk, config):
                self._on_speech(state, chunk, config)
            else:
                span = self._on_silence(state, chunk, config)

                if span is not None:
                    yield span


    def _is_speech(
        self,
        chunk: AudioChunk,
        config: PvcobraVadConfig,
    ) -> bool:
        
        voice_prob = self.handle.process(chunk.data)
        return voice_prob > config.threshold