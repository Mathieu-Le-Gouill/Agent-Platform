from unittest.mock import MagicMock, patch, mock_open
from uuid import uuid4

import pytest

from agent_platform.core.interfaces.loader.audio.config import AudioLoaderConfig
from agent_platform.core.schemas.enums import AudioFormat


class TestSoundFileLoader:
    @patch("agent_platform.integrations.loader.strategies.soundfile.soundfile.sf")
    @patch("builtins.open", new_callable=mock_open, read_data=b"fake_audio")
    async def test_load_wav(self, mock_file, mock_sf, mock_pil_image):
        from agent_platform.integrations.loader.strategies.soundfile.soundfile import (
            SoundFileLoader,
        )

        mock_info = MagicMock()
        mock_info.subtype = "FLOAT"
        mock_info.channels = 2
        mock_info.frames = 220500
        mock_sf.info.return_value = mock_info
        mock_data = MagicMock()
        mock_data.tobytes.return_value = b"fake_audio"
        mock_sf.read.return_value = (mock_data, 44100)

        loader = SoundFileLoader()
        results = await loader.load("/test/audio.wav")

        assert len(results) == 1
        doc = results[0]
        assert doc.source == "/test/audio.wav"
        assert doc.format == AudioFormat.WAV
        assert doc.sample_rate == 44100
        assert doc.channels == 2
        assert doc.subtype == "FLOAT"
        assert doc.content == b"fake_audio"

    @patch("agent_platform.integrations.loader.strategies.soundfile.soundfile.sf")
    @patch("builtins.open", new_callable=mock_open, read_data=b"data")
    async def test_load_mp3(self, mock_file, mock_sf, mock_pil_image):
        from agent_platform.integrations.loader.strategies.soundfile.soundfile import (
            SoundFileLoader,
        )

        mock_info = MagicMock()
        mock_info.subtype = "MP3"
        mock_info.channels = 1
        mock_info.frames = 44100
        mock_sf.info.return_value = mock_info
        mock_data = MagicMock()
        mock_data.tobytes.return_value = b"data"
        mock_sf.read.return_value = (mock_data, 44100)

        loader = SoundFileLoader()
        results = await loader.load("/test/song.mp3")

        assert len(results) == 1
        assert results[0].format == AudioFormat.MP3

    @patch("agent_platform.integrations.loader.strategies.soundfile.soundfile.sf")
    @patch("builtins.open", new_callable=mock_open, read_data=b"data")
    async def test_load_with_config(self, mock_file, mock_sf, mock_pil_image):
        from agent_platform.integrations.loader.strategies.soundfile.soundfile import (
            SoundFileLoader,
        )

        mock_info = MagicMock()
        mock_info.subtype = "FLOAT"
        mock_info.channels = 2
        mock_info.frames = 44100
        mock_sf.info.return_value = mock_info
        mock_data = MagicMock()
        mock_data.tobytes.return_value = b"fake_audio"
        mock_sf.read.return_value = (mock_data, 44100)

        loader = SoundFileLoader()
        config = AudioLoaderConfig(target_sample_rate=16000)
        results = await loader.load("/test/audio.wav", config=config)

        assert len(results) == 1
        assert results[0].sample_rate == 44100

    @patch("agent_platform.integrations.loader.strategies.soundfile.soundfile.sf")
    @patch("builtins.open", new_callable=mock_open, read_data=b"data")
    async def test_unknown_format(self, mock_file, mock_sf, mock_pil_image):
        from agent_platform.integrations.loader.strategies.soundfile.soundfile import (
            SoundFileLoader,
        )

        mock_info = MagicMock()
        mock_info.subtype = "UNKNOWN"
        mock_info.channels = 1
        mock_info.frames = 22050
        mock_sf.info.return_value = mock_info
        mock_data = MagicMock()
        mock_data.tobytes.return_value = b"data"
        mock_sf.read.return_value = (mock_data, 22050)

        loader = SoundFileLoader()
        results = await loader.load("/test/audio.xyz")

        assert len(results) == 1
        assert results[0].format == AudioFormat.UNKNOWN


class TestSoundFileLoaderDuration:
    @patch("agent_platform.integrations.loader.strategies.soundfile.soundfile.sf")
    @patch("builtins.open", new_callable=mock_open, read_data=b"data")
    async def test_duration_from_frames(self, mock_file, mock_sf, mock_pil_image):
        from agent_platform.integrations.loader.strategies.soundfile.soundfile import (
            SoundFileLoader,
        )

        mock_info = MagicMock()
        mock_info.subtype = "FLOAT"
        mock_info.channels = 2
        mock_info.frames = 220500
        mock_sf.info.return_value = mock_info
        mock_data = MagicMock()
        mock_data.tobytes.return_value = b"fake_audio"
        mock_sf.read.return_value = (mock_data, 44100)

        loader = SoundFileLoader()
        results = await loader.load("/test/audio.wav")

        assert results[0].duration == 5.0

    @patch("agent_platform.integrations.loader.strategies.soundfile.soundfile.sf")
    @patch("builtins.open", new_callable=mock_open, read_data=b"data")
    async def test_duration_none_when_zero_sample_rate(
        self, mock_file, mock_sf, mock_pil_image
    ):
        from agent_platform.integrations.loader.strategies.soundfile.soundfile import (
            SoundFileLoader,
        )

        mock_info = MagicMock()
        mock_info.subtype = "FLOAT"
        mock_info.channels = 1
        mock_info.frames = 0
        mock_sf.info.return_value = mock_info
        mock_data = MagicMock()
        mock_data.tobytes.return_value = b"data"
        mock_sf.read.return_value = (mock_data, 0)

        loader = SoundFileLoader()
        results = await loader.load("/test/silence.wav")

        assert results[0].duration is None
