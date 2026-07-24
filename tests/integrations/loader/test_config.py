from agent_platform.core.interfaces.loader.audio.config import AudioLoaderConfig
from agent_platform.core.interfaces.loader.config import LoaderConfig
from agent_platform.core.interfaces.loader.image.config import ImageLoaderConfig
from agent_platform.core.interfaces.loader.video.config import VideoLoaderConfig
from agent_platform.core.schemas.enums import ImageFormat
from agent_platform.integrations.loader.strategies.unstructured.config import (
    UnstructuredLoaderConfig,
)


class TestLoaderConfig:
    def test_defaults(self):
        cfg = LoaderConfig()
        assert cfg is not None

    def test_is_pydantic(self):
        from pydantic import BaseModel

        assert isinstance(LoaderConfig(), BaseModel)


class TestUnstructuredLoaderConfig:
    def test_defaults(self):
        cfg = UnstructuredLoaderConfig()
        assert cfg.mode == "elements"
        assert cfg.chunking_strategy is None

    def test_custom_mode(self):
        cfg = UnstructuredLoaderConfig(mode="single")
        assert cfg.mode == "single"

    def test_with_chunking_strategy(self):
        cfg = UnstructuredLoaderConfig(mode="elements", chunking_strategy="by_title")
        assert cfg.chunking_strategy == "by_title"


class TestImageLoaderConfig:
    def test_defaults(self):
        cfg = ImageLoaderConfig()
        assert cfg.target_format is None
        assert cfg.max_size is None

    def test_with_target_format(self):
        cfg = ImageLoaderConfig(target_format=ImageFormat.JPEG)
        assert cfg.target_format == ImageFormat.JPEG

    def test_with_max_size(self):
        cfg = ImageLoaderConfig(max_size=(800, 600))
        assert cfg.max_size == (800, 600)


class TestAudioLoaderConfig:
    def test_defaults(self):
        cfg = AudioLoaderConfig()
        assert cfg.target_sample_rate is None
        assert cfg.max_duration_sec is None

    def test_with_sample_rate(self):
        cfg = AudioLoaderConfig(target_sample_rate=16000)
        assert cfg.target_sample_rate == 16000

    def test_with_max_duration(self):
        cfg = AudioLoaderConfig(max_duration_sec=30.0)
        assert cfg.max_duration_sec == 30.0


class TestVideoLoaderConfig:
    def test_defaults(self):
        cfg = VideoLoaderConfig()
        assert cfg.extract_audio is False
        assert cfg.max_duration_sec is None

    def test_extract_audio_enabled(self):
        cfg = VideoLoaderConfig(extract_audio=True)
        assert cfg.extract_audio is True

    def test_with_max_duration(self):
        cfg = VideoLoaderConfig(max_duration_sec=60.0)
        assert cfg.max_duration_sec == 60.0
