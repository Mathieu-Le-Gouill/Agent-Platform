import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
    import webrtcvad

from agent_platform.core.interfaces.vad.framebased import FrameBasedVAD
from agent_platform.core.interfaces.vad.requirements import AudioRequirements
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.enums import DataType
from agent_platform.integrations.vad.webrtc.config import WebrtcVadConfig


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

    def _prepare(self, config: WebrtcVadConfig) -> None:
        self.model.set_mode(config.mode)

    def _is_speech(
        self,
        chunk: AudioChunk,
        config: WebrtcVadConfig,
    ) -> bool:

        return self.model.is_speech(
            chunk.data,
            config.sample_rate,
        )
