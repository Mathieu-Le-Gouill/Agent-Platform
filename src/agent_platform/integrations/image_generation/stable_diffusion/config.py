from __future__ import annotations
from agent_platform.core.interfaces.image_generation.config import ImageGenConfig


class StableDiffusionConfig(ImageGenConfig):
    model: str = "runwayml/stable-diffusion-v1-5"
    device: str = "cpu"
    dtype: str = "float32"
    safety_checker: bool = True
