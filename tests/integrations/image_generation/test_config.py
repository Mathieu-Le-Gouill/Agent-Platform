import pytest
from pydantic import ValidationError

from agent_platform.core.interfaces.image_generation.config import ImageGenConfig
from agent_platform.integrations.image_generation.dalle.config import DalleConfig
from agent_platform.integrations.image_generation.stable_diffusion.config import (
    StableDiffusionConfig,
)
from agent_platform.integrations.image_generation.midjourney.config import (
    MidjourneyConfig,
)


class TestImageGenConfig:
    def test_default_model(self):
        cfg = ImageGenConfig()
        assert cfg.model == "dall-e-3"

    def test_custom_model(self):
        cfg = ImageGenConfig(model="dall-e-2")
        assert cfg.model == "dall-e-2"


class TestDalleConfig:
    def test_inherits_defaults(self):
        cfg = DalleConfig()
        assert cfg.model == "dall-e-3"
        assert cfg.quality == "standard"

    def test_custom_quality(self):
        cfg = DalleConfig(quality="hd")
        assert cfg.quality == "hd"

    def test_invalid_quality_raises_validation_error(self):
        with pytest.raises(ValidationError):
            DalleConfig(quality="ultra")


class TestStableDiffusionConfig:
    def test_defaults(self):
        cfg = StableDiffusionConfig()
        assert cfg.model == "runwayml/stable-diffusion-v1-5"
        assert cfg.device == "cpu"
        assert cfg.dtype == "float32"
        assert cfg.safety_checker is True

    def test_custom_values(self):
        cfg = StableDiffusionConfig(
            model="stabilityai/stable-diffusion-2-1",
            device="cuda",
            dtype="float16",
            safety_checker=False,
        )
        assert cfg.model == "stabilityai/stable-diffusion-2-1"
        assert cfg.device == "cuda"
        assert cfg.dtype == "float16"
        assert cfg.safety_checker is False


class TestMidjourneyConfig:
    def test_default_api_url_and_timeout(self):
        cfg = MidjourneyConfig()
        assert cfg.api_url == "http://localhost:8080"
        assert cfg.timeout == 120.0

    def test_custom_api_url_via_alias(self):
        cfg = MidjourneyConfig(MIDJOURNEY_API_URL="https://mj.example.com")
        assert cfg.api_url == "https://mj.example.com"

    def test_custom_timeout(self):
        cfg = MidjourneyConfig(timeout=60.0)
        assert cfg.timeout == 60.0
