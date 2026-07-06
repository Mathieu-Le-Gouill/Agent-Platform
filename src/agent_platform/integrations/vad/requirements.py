from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from agent_platform.integrations.vad.configuration import VADConfig
from agent_platform.models.chunk import AudioChunk
from agent_platform.models.enums import DataType


T = TypeVar("T", bound="VADConfig")


@dataclass(frozen=True, slots=True)
class AudioRequirements:

    sample_rates: tuple[int, ...]
    channels: int
    dtype: DataType
    normalized: bool

    def validate(self, chunk: AudioChunk) -> None:
        if chunk.sample_rate not in self.sample_rates:
            raise ValueError(
                f"Unsupported sample rate {chunk.sample_rate}. "
                f"Expected one of {self.sample_rates}."
            )

        if chunk.channels != self.channels:
            raise ValueError(
                f"Expected {self.channels} channel(s), "
                f"got {chunk.channels}."
            )

        if chunk.dtype != self.dtype:
            raise TypeError(
                f"Expected dtype {self.dtype.__name__}, "
                f"got {chunk.dtype}."
            )
