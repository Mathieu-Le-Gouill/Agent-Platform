from unittest.mock import MagicMock, mock_open, patch

from agent_platform.core.schemas.enums import VideoFormat


class TestPyAVLoader:
    @patch("agent_platform.integrations.loader.strategies.pyav.pyav.av")
    @patch("builtins.open", new_callable=mock_open, read_data=b"fake_video_data")
    async def test_load_mp4_with_video_and_audio(
        self, mock_file, mock_av, mock_pil_image
    ):
        from agent_platform.integrations.loader.strategies.pyav.pyav import (
            PyAVLoader,
        )

        mock_video = MagicMock()
        mock_video.average_rate = 24.0
        mock_video.width = 1920
        mock_video.height = 1080
        mock_video.codec.name = "h264"
        mock_video.bit_rate = 5000000

        mock_audio = MagicMock()
        mock_audio.codec.name = "aac"
        mock_audio.channels = 2
        mock_audio.rate = 48000
        mock_audio.format.name = "fltp"

        mock_container = MagicMock()
        mock_container.streams.video = (mock_video,)
        mock_container.streams.audio = (mock_audio,)
        mock_container.duration = 30 * 1000000

        mock_av.open.return_value.__enter__.return_value = mock_container
        mock_av.time_base = 1000000

        loader = PyAVLoader()
        results = await loader.load("/test/video.mp4")

        assert len(results) == 1
        doc = results[0]
        assert doc.source == "/test/video.mp4"
        assert doc.format == VideoFormat.MP4
        assert doc.dimensions.width == 1920
        assert doc.dimensions.height == 1080
        assert doc.duration == 30.0
        assert doc.frame_rate == 24.0
        assert doc.codec == "h264"
        assert doc.bitrate == 5000000
        assert doc.has_audio is True
        assert doc.audio_codec == "aac"
        assert doc.audio_channels == 2
        assert doc.audio_sample_rate == 48000
        assert doc.content == b"fake_video_data"

    @patch("agent_platform.integrations.loader.strategies.pyav.pyav.av")
    @patch("builtins.open", new_callable=mock_open, read_data=b"data")
    async def test_load_returns_empty_when_no_video_stream(
        self, mock_file, mock_av, mock_pil_image
    ):
        from agent_platform.integrations.loader.strategies.pyav.pyav import (
            PyAVLoader,
        )

        mock_audio = MagicMock()

        mock_container = MagicMock()
        mock_container.streams.video = ()
        mock_container.streams.audio = (mock_audio,)
        mock_av.open.return_value.__enter__.return_value = mock_container

        loader = PyAVLoader()
        results = await loader.load("/test/audio_only.mp4")

        assert results == []

    @patch("agent_platform.integrations.loader.strategies.pyav.pyav.av")
    @patch("builtins.open", new_callable=mock_open, read_data=b"data")
    async def test_no_audio_stream(self, mock_file, mock_av, mock_pil_image):
        from agent_platform.integrations.loader.strategies.pyav.pyav import (
            PyAVLoader,
        )

        mock_video = MagicMock()
        mock_video.average_rate = 24.0
        mock_video.width = 640
        mock_video.height = 480
        mock_video.codec.name = "h264"
        mock_video.bit_rate = 1000000

        mock_container = MagicMock()
        mock_container.streams.video = (mock_video,)
        mock_container.streams.audio = ()
        mock_container.duration = None
        mock_av.open.return_value.__enter__.return_value = mock_container
        mock_av.time_base = 1000000

        loader = PyAVLoader()
        results = await loader.load("/test/video_no_audio.mp4")

        assert len(results) == 1
        assert results[0].has_audio is False
        assert results[0].audio_codec is None
        assert results[0].audio_channels is None
        assert results[0].duration is None

    @patch("agent_platform.integrations.loader.strategies.pyav.pyav.av")
    @patch("builtins.open", new_callable=mock_open, read_data=b"data")
    async def test_no_frame_rate(self, mock_file, mock_av, mock_pil_image):
        from agent_platform.integrations.loader.strategies.pyav.pyav import (
            PyAVLoader,
        )

        mock_video = MagicMock()
        mock_video.average_rate = None
        mock_video.width = 1280
        mock_video.height = 720
        mock_video.codec.name = "h265"
        mock_video.bit_rate = None

        mock_container = MagicMock()
        mock_container.streams.video = (mock_video,)
        mock_container.streams.audio = ()
        mock_container.duration = None
        mock_av.open.return_value.__enter__.return_value = mock_container
        mock_av.time_base = 1000000

        loader = PyAVLoader()
        results = await loader.load("/test/stream.ts")

        assert results[0].frame_rate is None
        assert results[0].duration is None
        assert results[0].bitrate is None
