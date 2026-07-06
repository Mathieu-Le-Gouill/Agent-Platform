from typing import Sequence, AsyncIterator
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
    import webrtcvad

from agent_platform.integrations.vad.framebased import FrameBasedVAD
from agent_platform.integrations.vad.configuration import WebrtcVadConfig
from agent_platform.models.chunk import AudioChunk
from agent_platform.models.span import SampleSpan
from agent_platform.integrations.vad.state import VADState
from agent_platform.integrations.vad.requirements import AudioRequirements
from agent_platform.models.enums import DataType
from agent_platform.audio.io import AudioIO


class Webrtcvad(FrameBasedVAD[WebrtcVadConfig]):
    def __init__(self):
        self.model = webrtcvad.Vad()

    @property
    def requirements(self) -> AudioRequirements:
        return AudioRequirements(
            sample_rates=(8000, 16000, 32000, 48000),
            channels=1,
            dtype=DataType.INT16,
            normalized=False,
        )

    def detect(
        self,
        audio_sequence: Sequence[AudioChunk],
        config: WebrtcVadConfig | None = None,
    ) -> list[SampleSpan]:

        if not audio_sequence:
            return []

        config = config or WebrtcVadConfig()
        state = VADState()

        self.model.set_mode(config.mode)

        voiced_frames: list[SampleSpan] = []

        for chunk in audio_sequence:
            if chunk.sample_rate != config.sample_rate:
                raise ValueError(
                    f"Expected {config.sample_rate}, got {chunk.sample_rate}"
                )
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
        config: WebrtcVadConfig | None = None,
    ) -> AsyncIterator[SampleSpan]:

        config = config or WebrtcVadConfig()
        state = VADState()

        self.model.set_mode(config.mode)

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
        config: WebrtcVadConfig,
    ) -> bool:

        return self.model.is_speech(
            AudioIO.to_numpy(chunk),
            config.sample_rate,
        )
