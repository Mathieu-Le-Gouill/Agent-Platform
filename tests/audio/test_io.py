import base64
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from agent_platform.audio.io import AudioIO, _from_numpy_dtype, _to_numpy_dtype
from agent_platform.models.chunk import AudioChunk
from agent_platform.models.document import AudioDocument
from agent_platform.models.enums import AudioFormat, DataType


class TestToNumpyDtype:
    def test_int16(self):
        assert _to_numpy_dtype(DataType.INT16) == np.int16

    def test_float32(self):
        assert _to_numpy_dtype(DataType.FLOAT32) == np.float32

    def test_int8(self):
        assert _to_numpy_dtype(DataType.INT8) == np.int8

    def test_uint8(self):
        assert _to_numpy_dtype(DataType.UINT8) == np.uint8


class TestFromNumpyDtype:
    def test_int16(self):
        assert _from_numpy_dtype(np.int16) == DataType.INT16

    def test_int8(self):
        assert _from_numpy_dtype(np.int8) == DataType.INT8

    def test_uint8(self):
        assert _from_numpy_dtype(np.uint8) == DataType.UINT8

    def test_float32(self):
        assert _from_numpy_dtype(np.float32) == DataType.FLOAT32

    def test_float64_maps_to_float32(self):
        assert _from_numpy_dtype(np.float64) == DataType.FLOAT32

    def test_unsupported_raises(self):
        with pytest.raises(ValueError, match="Unsupported numpy dtype"):
            _from_numpy_dtype(np.int32)

    def test_unsupported_int64(self):
        with pytest.raises(ValueError, match="Unsupported numpy dtype"):
            _from_numpy_dtype(np.int64)


class TestAudioIO:
    def test_from_tensor_1d(self):
        tensor = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        chunk = AudioIO.from_tensor(tensor, sample_rate=16000)
        assert chunk.sample_rate == 16000
        assert chunk.channels == 1
        assert chunk.dtype == DataType.FLOAT32
        assert isinstance(chunk.data, bytes)

    def test_from_tensor_2d(self):
        tensor = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
        chunk = AudioIO.from_tensor(tensor, sample_rate=44100)
        assert chunk.channels == 2
        assert chunk.sample_rate == 44100

    def test_from_tensor_accepts_kwargs(self):
        tensor = torch.tensor([0.1, 0.2], dtype=torch.float32)
        chunk = AudioIO.from_tensor(tensor, sample_rate=16000, start=0, end=32000)
        assert chunk.start == 0
        assert chunk.end == 32000

    def test_from_tensor_promotes_to_float32(self):
        tensor = torch.tensor([1, 2, 3], dtype=torch.int16)
        chunk = AudioIO.from_tensor(tensor, sample_rate=16000)
        assert chunk.dtype == DataType.FLOAT32

    def test_to_tensor_mono(self):
        chunk = AudioChunk(
            data=bytes([0x00, 0x00, 0x80, 0x3F]),
            sample_rate=16000,
            channels=1,
            dtype=DataType.FLOAT32,
        )
        tensor = AudioIO.to_tensor(chunk)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        assert tensor.shape == (1,)

    def test_to_tensor_stereo(self):
        data = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32).tobytes()
        chunk = AudioChunk(
            data=data, sample_rate=44100, channels=2, dtype=DataType.FLOAT32
        )
        tensor = AudioIO.to_tensor(chunk)
        assert tensor.shape == (2, 2)

    def test_to_numpy(self):
        chunk = AudioChunk(
            data=bytes([0x00, 0x00, 0x80, 0x3F]),
            sample_rate=16000,
            channels=1,
            dtype=DataType.FLOAT32,
        )
        arr = AudioIO.to_numpy(chunk)
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        assert arr.shape == (1,)

    def test_to_numpy_stereo(self):
        data = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32).tobytes()
        chunk = AudioChunk(
            data=data, sample_rate=44100, channels=2, dtype=DataType.FLOAT32
        )
        arr = AudioIO.to_numpy(chunk)
        assert arr.shape == (2, 2)

    def test_from_base64(self):
        original = b"\x00\x01\x02\x03"
        encoded = base64.b64encode(original).decode("utf-8")
        chunk = AudioIO.from_base64(encoded, sample_rate=16000)
        assert chunk.data == original
        assert chunk.sample_rate == 16000

    def test_from_base64_accepts_kwargs(self):
        encoded = base64.b64encode(b"\x00\x01").decode("utf-8")
        chunk = AudioIO.from_base64(
            encoded, sample_rate=44100, channels=2, dtype=DataType.INT16
        )
        assert chunk.channels == 2
        assert chunk.dtype == DataType.INT16

    def test_to_base64(self):
        data = b"\x00\x01\x02\x03"
        chunk = AudioChunk(
            data=data, sample_rate=16000, channels=1, dtype=DataType.FLOAT32
        )
        result = AudioIO.to_base64(chunk)
        expected = base64.b64encode(data).decode("utf-8")
        assert result == expected

    def test_from_base64_roundtrip(self):
        original = b"\xab\xcd\xef\x01"
        encoded = base64.b64encode(original).decode("utf-8")
        chunk = AudioIO.from_base64(encoded, sample_rate=16000)
        result = AudioIO.to_base64(chunk)
        assert result == encoded

    def test_to_tensor_roundtrip(self):
        tensor = torch.tensor([0.5, 0.25, -0.1], dtype=torch.float32)
        chunk = AudioIO.from_tensor(tensor, sample_rate=16000)
        result = AudioIO.to_tensor(chunk)
        assert torch.allclose(tensor, result, atol=1e-6)

    def test_to_numpy_roundtrip(self):
        arr = np.array([0.5, 0.25, -0.1], dtype=np.float32)
        tensor = torch.from_numpy(arr)
        chunk = AudioIO.from_tensor(tensor, sample_rate=16000)
        result = AudioIO.to_numpy(chunk)
        np.testing.assert_array_almost_equal(arr, result)

    @patch("soundfile.read")
    @patch("soundfile.info")
    def test_from_file(self, mock_info, mock_read):
        mock_info.return_value = MagicMock(
            format="wav", channels=1, frames=16000, subtype="PCM_16"
        )
        mock_read.return_value = (np.array([[0.1, 0.2]], dtype=np.float32), 16000)

        doc = AudioIO.from_file("/fake/path.wav")
        assert doc.source == "/fake/path.wav"
        assert doc.format == AudioFormat.WAV
        assert doc.sample_rate == 16000
        assert doc.channels == 1
        assert doc.duration == 1.0
        assert doc.subtype == "PCM_16"

    @patch("soundfile.read")
    @patch("soundfile.info")
    def test_from_file_unknown_format(self, mock_info, mock_read):
        mock_info.return_value = MagicMock(
            format=None, channels=1, frames=16000, subtype=None
        )
        mock_read.return_value = (np.array([[0.1]], dtype=np.float32), 16000)

        doc = AudioIO.from_file("/fake/path.xyz")
        assert doc.format == AudioFormat.UNKNOWN

    @patch("soundfile.read")
    @patch("soundfile.info")
    def test_from_file_zero_frames(self, mock_info, mock_read):
        mock_info.return_value = MagicMock(
            format="wav", channels=1, frames=0, subtype="PCM_16"
        )
        mock_read.return_value = (np.array([[]], dtype=np.float32), 0)

        doc = AudioIO.from_file("/fake/path.wav")
        assert doc.duration is None

    def test_from_bytes(self):
        chunk = AudioIO.from_bytes(b"\x00\x01\x02\x03", sample_rate=16000)
        assert chunk.data == b"\x00\x01\x02\x03"
        assert chunk.sample_rate == 16000
        assert chunk.channels == 1

    def test_from_bytes_accepts_kwargs(self):
        chunk = AudioIO.from_bytes(
            b"\x00\x01", sample_rate=44100, channels=2, dtype=DataType.INT16
        )
        assert chunk.channels == 2
        assert chunk.dtype == DataType.INT16
