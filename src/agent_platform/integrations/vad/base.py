from __future__ import annotations
from typing import Generic, TypeVar, Sequence, AsyncIterator

from abc import ABC, abstractmethod
from agent_platform.integrations.vad.configuration import VADConfig
from agent_platform.models.span import SampleSpan
from agent_platform.models.chunk import AudioChunk
from agent_platform.integrations.vad.requirements import AudioRequirements

T = TypeVar("T", bound="VADConfig")
    
class BaseVAD(ABC, Generic[T]):

    @property
    @abstractmethod
    def requirements(self) -> AudioRequirements:
        ...
        

    @abstractmethod
    def detect(
        self, 
        audio_sequence: Sequence[AudioChunk],
        config: T | None,
    ) -> list[SampleSpan]: 
        ...


    @abstractmethod
    def adetect(
        self, 
        audio_sequence: AsyncIterator[AudioChunk],
        config: T | None,
    ) -> AsyncIterator[SampleSpan]: 
        ...


    @abstractmethod
    def _is_speech(
        self, 
        chunk: AudioChunk, 
        config: T,
    ) -> bool:
        ...