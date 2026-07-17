from __future__ import annotations
from pydantic import Field, model_validator
from agent_platform.core.interfaces.image_generation.config import ImageGenConfig


class DalleConfig(ImageGenConfig):
    quality: str = Field(default="standard", pattern=r"^(standard|hd)$")
