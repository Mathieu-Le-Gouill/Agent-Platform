from __future__ import annotations

import io
import torch
import torchaudio
from uuid import uuid4

from agent_platform.models.document import AudioDocument
from agent_platform.models.chunk import AudioChunk
from agent_platform.models.enums import DataType
from agent_platform.audio.io import AudioIO


class AudioDSP:

    @staticmethod
    def decode(audio: AudioDocument) -> tuple[torch.Tensor, int]:
        buffer = io.BytesIO(audio.content)
        waveform, sr = torchaudio.load(buffer)
        return waveform, sr

    @staticmethod
    def chunk_waveform(
        waveform: torch.Tensor,
        sample_rate: int,
        chunk_duration_ms: int = 10_000,
    ) -> list[torch.Tensor]:
        chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        chunks = []
        total_len = waveform.shape[-1]
        for i in range(0, total_len, chunk_size):
            chunks.append(waveform[:, i:i + chunk_size])
        return chunks

    @staticmethod
    def build_chunks(
        waveform: torch.Tensor,
        audio: AudioDocument,
        chunk_duration_ms: int = 10_000,
    ) -> list[AudioChunk]:
        sr = audio.sample_rate
        if sr is None:
            raise ValueError("sample_rate is required")

        chunk_size = int(sr * chunk_duration_ms / 1000)
        result: list[AudioChunk] = []
        total_len = waveform.shape[-1]
        start_sample = 0
        chunk_index = 0

        for i in range(0, total_len, chunk_size):
            chunk = waveform[:, i:i + chunk_size]
            end_sample = i + chunk.shape[-1]

            result.append(
                AudioIO.from_tensor(
                    chunk,
                    sample_rate=sr,
                    id=uuid4(),
                    index=chunk_index,
                    document_id=audio.id,
                    start=start_sample,
                    end=end_sample,
                    format=audio.format,
                    dtype=DataType.FLOAT32,
                )
            )

            start_sample = end_sample
            chunk_index += 1

        return result

    @staticmethod
    def resample(chunk: AudioChunk, target_rate: int) -> AudioChunk:
        if chunk.sample_rate == target_rate:
            return chunk

        waveform = AudioIO.to_tensor(chunk)
        waveform = torchaudio.functional.resample(waveform, chunk.sample_rate, target_rate)

        return AudioIO.from_tensor(
            waveform,
            sample_rate=target_rate,
            id=chunk.id,
            index=chunk.index,
            document_id=chunk.document_id,
            channels=waveform.shape[0],
            start=chunk.start,
            end=chunk.end,
            format=chunk.format,
            dtype=chunk.dtype,
        )

    @staticmethod
    def process(
        audio: AudioDocument,
        chunk_duration_ms: int = 10_000,
        target_sample_rate: int | None = None,
    ) -> list[AudioChunk]:
        waveform, sr = AudioDSP.decode(audio)

        if target_sample_rate and target_sample_rate != sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sample_rate)
            sr = target_sample_rate

        return AudioDSP.build_chunks(waveform, audio, chunk_duration_ms)
