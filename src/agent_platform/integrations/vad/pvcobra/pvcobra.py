from __future__ import annotations

import pvcobra
from typing import AsyncIterator, Sequence
from agent_platform.integrations.credentials import PicoVoiceCredentials
from agent_platform.core.interfaces.vad.framebased import FrameBasedVAD
from agent_platform.integrations.vad.pvcobra.config import PvcobraVadConfig
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.span import SampleSpan
from agent_platform.core.interfaces.vad.state import VADState
from agent_platform.core.interfaces.vad.requirements import AudioRequirements
from agent_platform.core.schemas.enums import DataType
from agent_platform.core.errors import ProviderError


class PvcobraVAD(FrameBasedVAD[PvcobraVadConfig]):
    def __init__(self, credentials: PicoVoiceCredentials | None = None) -> None:
        self._credentials = (
            credentials if credentials is not None else PicoVoiceCredentials()
        )
        self._handle: pvcobra.Cobra | None = None

    def _default_config(self) -> PvcobraVadConfig:
        return PvcobraVadConfig()

    def _ensure_handle(self, config: PvcobraVadConfig) -> pvcobra.Cobra:
        if self._handle is None:
            key = (
                self._credentials.access_key.get_secret_value()
                if self._credentials.access_key
                else ""
            )
            self._handle = pvcobra.create(key, config.device, config.library_path)
        return self._handle

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.delete()
            finally:
                self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

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
        config: PvcobraVadConfig | None = None,
    ) -> list[SampleSpan]:

        config = config or self._default_config()
        self._ensure_handle(config)
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

        config = config or self._default_config()
        self._ensure_handle(config)
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
        if self._handle is None:
            raise RuntimeError("VAD handle not initialized. Call detect() first.")

        expected_samples = self._handle.frame_length
        num_samples = len(chunk.data) // 2  # int16

        if num_samples != expected_samples:
            raise ProviderError(
                f"Cobra requires frames of exactly {expected_samples} samples, "
                f"got {num_samples} samples"
            )

        try:
            voice_prob = self._handle.process(chunk.data)
        except Exception as exc:
            raise ProviderError(
                f"Cobra voice activity processing failed: {exc}"
            ) from exc

        return voice_prob > config.threshold


# Ref: https://picovoice.ai/docs/api/cobra-python/
