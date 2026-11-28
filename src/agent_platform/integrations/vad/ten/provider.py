from __future__ import annotations

from ten_vad import TenVad

from agent_platform.audio.io import AudioIO
from agent_platform.core.interfaces.vad.framebased import FrameBasedVAD
from agent_platform.core.interfaces.vad.requirements import AudioRequirements
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.enums import DataType
from agent_platform.integrations.vad.ten.config import TenVadConfig


class TenVAD(FrameBasedVAD[TenVadConfig]):
    def __init__(self) -> None:
        self.handle: TenVad | None = None
        self._handle_config: tuple[int, float] | None = None

    def _default_config(self) -> TenVadConfig:
        return TenVadConfig()

    def _ensure_handle(self, config: TenVadConfig) -> TenVad:
        key = (config.hop_size, config.threshold)

        if self.handle is None or self._handle_config != key:
            self.handle = TenVad(config.hop_size, config.threshold)
            self._handle_config = key

        return self.handle

    @property
    def requirements(self) -> AudioRequirements:
        return AudioRequirements(
            sample_rates=(16000,),
            channels=1,
            dtype=DataType.INT16,
            normalized=False,
        )

    def _prepare(self, config: TenVadConfig) -> None:
        self._ensure_handle(config)

    def _is_speech(
        self,
        chunk: AudioChunk,
        config: TenVadConfig,
    ) -> bool:

        handle = self._ensure_handle(config)
        voice_prob, flag = handle.process(AudioIO.to_numpy(chunk))
        return voice_prob > config.threshold


# Ref: https://github.com/TEN-framework/ten-vad
