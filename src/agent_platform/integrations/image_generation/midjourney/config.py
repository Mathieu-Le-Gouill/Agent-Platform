from __future__ import annotations
from pydantic import Field
from agent_platform.core.interfaces.image_generation.config import ImageGenConfig


class MidjourneyConfig(ImageGenConfig):
    api_url: str = Field(default="http://localhost:8080", alias="MIDJOURNEY_API_URL")
    timeout: int = 120
