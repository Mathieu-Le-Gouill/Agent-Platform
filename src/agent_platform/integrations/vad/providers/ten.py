from __future__ import annotations

from ten_vad import TenVad
from typing import AsyncIterator, Sequence

from agent_platform.integrations.vad.framebased import FrameBasedVAD
from agent_platform.integrations.vad.configuration import TenVadConfig
from agent_platform.models.chunk import AudioChunk
from agent_platform.models.span import SampleSpan
from agent_platform.integrations.vad.state import VADState
from agent_platform.audio.io import AudioIO
from agent_platform.integrations.vad.requirements import AudioRequirements
from agent_platform.models.enums import DataType


class TenVAD(FrameBasedVAD[TenVadConfig]):
    @property
    def requirements(self) -> AudioRequirements:
        return AudioRequirements(
            sample_rates=(16000,),
            channels=1,
            dtype=DataType.INT16,
            normalized=True,
        )

    def detect(
        self, audio_sequence: Sequence[AudioChunk], config: TenVadConfig | None = None
    ) -> list[SampleSpan]:

        config = config or TenVadConfig()
        state = VADState()

        self.handle = TenVad(config.hop_size, config.threshold)

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
        config: TenVadConfig | None = None,
    ) -> AsyncIterator[SampleSpan]:

        config = config or TenVadConfig()
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
        config: TenVadConfig,
    ) -> bool:

        voice_prob, flag = self.handle.process(AudioIO.to_numpy(chunk))
        return voice_prob > config.threshold
