from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Sequence, AsyncIterator

from agent_platform.core.interfaces.vad.config import VADConfig
from agent_platform.core.schemas.span import SampleSpan
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.interfaces.vad.requirements import AudioRequirements

ConfigT = TypeVar("ConfigT", bound="VADConfig")


class BaseVAD(ABC, Generic[ConfigT]):
    @property
    def requirements(self) -> AudioRequirements: ...

    @abstractmethod
    def detect(
        self,
        audio_sequence: Sequence[AudioChunk],
        config: ConfigT | None,
    ) -> list[SampleSpan]: ...

    @abstractmethod
    def adetect(
        self,
        audio_sequence: AsyncIterator[AudioChunk],
        config: ConfigT | None,
    ) -> AsyncIterator[SampleSpan]: ...

    @abstractmethod
    def _is_speech(
        self,
        chunk: AudioChunk,
        config: ConfigT,
    ) -> bool: ...

    def _validate_chunk(
        self,
        chunk: AudioChunk,
        config_sample_rate: int,
    ) -> None:

        self.requirements.validate(chunk)

        if chunk.sample_rate != config_sample_rate:
            raise ValueError(
                f"configuration sample rate differ from chunk sample rate, got {chunk.sample_rate}, expected {config_sample_rate}"
            )
