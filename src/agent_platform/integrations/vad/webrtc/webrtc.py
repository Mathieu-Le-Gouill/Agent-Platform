from typing import Sequence, AsyncIterator
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
    import webrtcvad

from agent_platform.core.interfaces.vad.framebased import FrameBasedVAD
from agent_platform.integrations.vad.webrtc.config import WebrtcVadConfig
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.span import SampleSpan
from agent_platform.core.interfaces.vad.state import VADState
from agent_platform.core.interfaces.vad.requirements import AudioRequirements
from agent_platform.core.schemas.enums import DataType
from agent_platform.audio.io import AudioIO


class Webrtcvad(FrameBasedVAD[WebrtcVadConfig]):
    _VALID_FRAME_DURATIONS_MS = (10, 20, 30)
    _BYTES_PER_SAMPLE = 2  # int16

    def __init__(self) -> None:
        self.model = webrtcvad.Vad()

    def _default_config(self) -> WebrtcVadConfig:
        return WebrtcVadConfig()

    def _validate_chunk(
        self,
        chunk: AudioChunk,
        config_sample_rate: int,
    ) -> None:
        super()._validate_chunk(chunk, config_sample_rate)

        num_samples = len(chunk.data) // self._BYTES_PER_SAMPLE
        valid_sample_counts = {
            config_sample_rate * ms // 1000 for ms in self._VALID_FRAME_DURATIONS_MS
        }

        if num_samples not in valid_sample_counts:
            raise ValueError(
                "webrtcvad requires 10/20/30ms frames at "
                f"{config_sample_rate}Hz ({sorted(valid_sample_counts)} samples), "
                f"got {num_samples} samples"
            )

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

        config = config or self._default_config()
        state = VADState()

        self.model.set_mode(config.mode)

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
        config: WebrtcVadConfig | None = None,
    ) -> AsyncIterator[SampleSpan]:

        config = config or self._default_config()
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
