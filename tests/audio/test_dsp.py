from __future__ import annotations

import io
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
import torch

from agent_platform.audio.dsp import AudioDSP
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.document import AudioDocument
from agent_platform.core.schemas.enums import AudioFormat, DataType


class TestDecode:
    @patch("torchaudio.load")
    def test_decode(self, mock_load):
        mock_waveform = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
        mock_load.return_value = (mock_waveform, 16000)

        audio = AudioDocument(content=b"fake_bytes")
        waveform, sr = AudioDSP.decode(audio)

        assert torch.equal(waveform, mock_waveform)
        assert sr == 16000
        mock_load.assert_called_once()
        args, _ = mock_load.call_args
        assert isinstance(args[0], io.BytesIO)
        assert args[0].getvalue() == b"fake_bytes"


class TestChunkWaveform:
    def test_even_division(self):
        waveform = torch.randn(1, 32000)
        chunks = AudioDSP.chunk_waveform(waveform, 16000, chunk_duration_ms=1000)
        assert len(chunks) == 2
        assert chunks[0].shape[-1] == 16000
        assert chunks[1].shape[-1] == 16000
        assert torch.equal(chunks[0], waveform[:, :16000])
        assert torch.equal(chunks[1], waveform[:, 16000:])

    def test_with_remainder(self):
        waveform = torch.randn(1, 32100)
        chunks = AudioDSP.chunk_waveform(waveform, 16000, chunk_duration_ms=1000)
        assert len(chunks) == 3
        assert chunks[2].shape[-1] == 100

    def test_single_chunk(self):
        waveform = torch.randn(2, 8000)
        chunks = AudioDSP.chunk_waveform(waveform, 16000, chunk_duration_ms=1000)
        assert len(chunks) == 1
        assert chunks[0].shape == (2, 8000)

    def test_empty_waveform(self):
        waveform = torch.randn(1, 0)
        chunks = AudioDSP.chunk_waveform(waveform, 16000)
        assert len(chunks) == 0

    def test_stereo_preserves_channels(self):
        waveform = torch.randn(2, 48000)
        chunks = AudioDSP.chunk_waveform(waveform, 16000, chunk_duration_ms=1000)
        assert len(chunks) == 3
        for ch in chunks:
            assert ch.shape[0] == 2

    def test_default_chunk_duration(self):
        waveform = torch.randn(1, 16000 * 10)
        chunks = AudioDSP.chunk_waveform(waveform, 16000)
        assert len(chunks) == 1


class TestBuildChunks:
    @patch("agent_platform.audio.dsp.AudioIO.from_tensor")
    def test_metadata_propagation(self, mock_from_tensor):
        doc_id = uuid4()
        audio = AudioDocument(
            id=doc_id,
            content=b"",
            sample_rate=16000,
            format=AudioFormat.WAV,
        )
        mock_from_tensor.return_value = MagicMock(spec=AudioChunk)

        _ = AudioDSP.build_chunks(torch.randn(1, 32000), audio, chunk_duration_ms=1000)

        assert mock_from_tensor.call_count == 2
        for call_args in mock_from_tensor.call_args_list:
            _, kwargs = call_args
            assert kwargs["sample_rate"] == 16000
            assert kwargs["document_id"] == doc_id
            assert kwargs["format"] == AudioFormat.WAV
            assert kwargs["dtype"] == DataType.FLOAT32

    def test_raises_when_no_sample_rate(self):
        audio = AudioDocument(content=b"", sample_rate=None)
        with pytest.raises(ValueError, match="sample_rate is required"):
            AudioDSP.build_chunks(torch.randn(1, 16000), audio)

    @patch("agent_platform.audio.dsp.AudioIO.from_tensor")
    def test_chunk_indices_and_bounds(self, mock_from_tensor):
        audio = AudioDocument(content=b"", sample_rate=16000)
        mock_from_tensor.return_value = MagicMock(spec=AudioChunk)

        AudioDSP.build_chunks(torch.randn(1, 32000), audio, chunk_duration_ms=1000)

        first = mock_from_tensor.call_args_list[0][1]
        assert first["index"] == 0
        assert first["start"] == 0
        assert first["end"] == 16000

        second = mock_from_tensor.call_args_list[1][1]
        assert second["index"] == 1
        assert second["start"] == 16000
        assert second["end"] == 32000

    @patch("agent_platform.audio.dsp.AudioIO.from_tensor")
    def test_partial_final_chunk(self, mock_from_tensor):
        audio = AudioDocument(content=b"", sample_rate=16000)
        mock_from_tensor.return_value = MagicMock(spec=AudioChunk)

        AudioDSP.build_chunks(torch.randn(1, 16100), audio, chunk_duration_ms=1000)

        second = mock_from_tensor.call_args_list[1][1]
        assert second["start"] == 16000
        assert second["end"] == 16100


class TestResample:
    def test_same_rate_returns_same_chunk(self):
        chunk = AudioChunk(
            data=b"\x00\x00\x80?",
            sample_rate=16000,
            channels=1,
            dtype=DataType.FLOAT32,
        )
        result = AudioDSP.resample(chunk, 16000)
        assert result is chunk

    @patch("agent_platform.audio.dsp.torchaudio.functional.resample")
    @patch("agent_platform.audio.dsp.AudioIO.to_tensor")
    @patch("agent_platform.audio.dsp.AudioIO.from_tensor")
    def test_different_rate(self, mock_from_tensor, mock_to_tensor, mock_resample):
        chunk = AudioChunk(
            data=b"\x00\x00\x80?",
            sample_rate=16000,
            channels=1,
            dtype=DataType.FLOAT32,
            index=0,
            document_id=uuid4(),
            start=0,
            end=16000,
            format=AudioFormat.WAV,
        )
        mock_waveform = torch.tensor([[0.5]], dtype=torch.float32)
        mock_resampled = torch.tensor([[0.5]], dtype=torch.float32)
        mock_to_tensor.return_value = mock_waveform
        mock_resample.return_value = mock_resampled
        mock_from_tensor.return_value = MagicMock(spec=AudioChunk)

        AudioDSP.resample(chunk, 44100)

        mock_to_tensor.assert_called_once_with(chunk)
        mock_resample.assert_called_once_with(mock_waveform, 16000, 44100)
        _, kwargs = mock_from_tensor.call_args
        assert kwargs["sample_rate"] == 44100
        assert kwargs["id"] == chunk.id
        assert kwargs["index"] == chunk.index
        assert kwargs["document_id"] == chunk.document_id
        assert kwargs["start"] == chunk.start
        assert kwargs["end"] == chunk.end
        assert kwargs["format"] == chunk.format
        assert kwargs["dtype"] == chunk.dtype


class TestProcess:
    @patch.object(AudioDSP, "decode")
    @patch.object(AudioDSP, "build_chunks")
    def test_default(self, mock_build_chunks, mock_decode):
        audio = MagicMock(spec=AudioDocument)
        mock_waveform = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
        mock_decode.return_value = (mock_waveform, 16000)
        mock_build_chunks.return_value = [MagicMock(spec=AudioChunk)]

        result = AudioDSP.process(audio)

        mock_decode.assert_called_once_with(audio)
        mock_build_chunks.assert_called_once_with(mock_waveform, audio, 10000)
        assert len(result) == 1

    @patch.object(AudioDSP, "decode")
    @patch.object(AudioDSP, "build_chunks")
    @patch("agent_platform.audio.dsp.torchaudio.functional.resample")
    def test_with_target_rate(self, mock_resample, mock_build_chunks, mock_decode):
        audio = MagicMock(spec=AudioDocument)
        mock_waveform = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
        mock_resampled = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
        mock_decode.return_value = (mock_waveform, 16000)
        mock_resample.return_value = mock_resampled
        mock_build_chunks.return_value = [MagicMock(spec=AudioChunk)]

        AudioDSP.process(audio, target_sample_rate=44100)

        mock_resample.assert_called_once_with(mock_waveform, 16000, 44100)
        mock_build_chunks.assert_called_once_with(mock_resampled, audio, 10000)

    @patch.object(AudioDSP, "decode")
    @patch.object(AudioDSP, "build_chunks")
    def test_target_rate_matches_source(self, mock_build_chunks, mock_decode):
        audio = MagicMock(spec=AudioDocument)
        mock_waveform = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
        mock_decode.return_value = (mock_waveform, 16000)
        mock_build_chunks.return_value = [MagicMock(spec=AudioChunk)]

        AudioDSP.process(audio, target_sample_rate=16000)

        mock_build_chunks.assert_called_once_with(mock_waveform, audio, 10000)

    @patch.object(AudioDSP, "decode")
    @patch.object(AudioDSP, "build_chunks")
    def test_custom_chunk_duration(self, mock_build_chunks, mock_decode):
        audio = MagicMock(spec=AudioDocument)
        mock_waveform = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
        mock_decode.return_value = (mock_waveform, 16000)
        mock_build_chunks.return_value = [MagicMock(spec=AudioChunk)]

        AudioDSP.process(audio, chunk_duration_ms=5000)

        mock_build_chunks.assert_called_once_with(mock_waveform, audio, 5000)
