import pytest
from pydantic import ValidationError

from agent_platform.core.interfaces.image_generation.config import ImageGenConfig
from agent_platform.integrations.image_generation.dalle.config import DalleConfig
from agent_platform.integrations.image_generation.midjourney.config import (
    MidjourneyConfig,
)
from agent_platform.integrations.image_generation.stable_diffusion.config import (
    StableDiffusionConfig,
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
        assert cfg.style is None

    def test_custom_quality(self):
        cfg = DalleConfig(quality="hd")
        assert cfg.quality == "hd"

    def test_invalid_quality_raises_validation_error(self):
        with pytest.raises(ValidationError):
            DalleConfig(quality="ultra")

    def test_style_field_dalle3(self):
        cfg = DalleConfig(style="vivid")
        assert cfg.style == "vivid"

    def test_invalid_style_raises_validation_error(self):
        with pytest.raises(ValidationError):
            DalleConfig(style="dramatic")

    def test_gpt_image_1_valid_quality(self):
        cfg = DalleConfig(model="gpt-image-1", quality="high")
        assert cfg.quality == "high"
        for quality in ("medium", "low", "auto"):
            assert DalleConfig(model="gpt-image-1", quality=quality).quality == quality

    def test_gpt_image_1_invalid_quality_raises(self):
        with pytest.raises(ValidationError):
            DalleConfig(model="gpt-image-1", quality="standard")

    def test_dalle3_invalid_quality_for_gpt_vocab_raises(self):
        with pytest.raises(ValidationError):
            DalleConfig(model="dall-e-3", quality="high")

    def test_dalle2_ignores_quality_validation(self):
        cfg = DalleConfig(model="dall-e-2", quality="anything")
        assert cfg.quality == "anything"


class TestStableDiffusionConfig:
    def test_defaults(self):
        cfg = StableDiffusionConfig()
        assert cfg.model == "runwayml/stable-diffusion-v1-5"
        assert cfg.device == "cpu"
        assert cfg.dtype == "float32"
        assert cfg.safety_checker is True
        assert cfg.guidance_scale == 7.5
        assert cfg.num_inference_steps == 50
        assert cfg.negative_prompt is None
        assert cfg.seed is None

    def test_custom_values(self):
        cfg = StableDiffusionConfig(
            model="stabilityai/stable-diffusion-2-1",
            device="cuda",
            dtype="float16",
            safety_checker=False,
            guidance_scale=9.0,
            num_inference_steps=30,
            negative_prompt="blurry, low quality",
            seed=42,
        )
        assert cfg.model == "stabilityai/stable-diffusion-2-1"
        assert cfg.device == "cuda"
        assert cfg.dtype == "float16"
        assert cfg.safety_checker is False
        assert cfg.guidance_scale == 9.0
        assert cfg.num_inference_steps == 30
        assert cfg.negative_prompt == "blurry, low quality"
        assert cfg.seed == 42


class TestMidjourneyConfig:
    def test_default_api_url_and_timeout(self):
        cfg = MidjourneyConfig()
        assert cfg.api_url == "http://localhost:8080"
        assert cfg.timeout == 120.0
        assert cfg.process_mode == "fast"

    def test_custom_api_url_via_alias(self):
        cfg = MidjourneyConfig(MIDJOURNEY_API_URL="https://mj.example.com")
        assert cfg.api_url == "https://mj.example.com"

    def test_custom_timeout(self):
        cfg = MidjourneyConfig(timeout=60.0)
        assert cfg.timeout == 60.0

    def test_custom_process_mode(self):
        for mode in ("fast", "relax", "turbo"):
            assert MidjourneyConfig(process_mode=mode).process_mode == mode

    def test_invalid_process_mode_raises_validation_error(self):
        with pytest.raises(ValidationError):
            MidjourneyConfig(process_mode="ludicrous")
