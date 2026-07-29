from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent_platform.agents.tools._utils import safe_call
from agent_platform.agents.tools.base import Tool
from agent_platform.core.interfaces.image_generation.base import BaseImageGenerator
from agent_platform.core.interfaces.image_generation.config import ImageGenConfig
from agent_platform.core.schemas.document import ImageDocument
from agent_platform.core.schemas.enums import ImageFormat


class GenerateImageInput(BaseModel):
    prompt: str = Field(
        ..., min_length=1, description="Text prompt describing the image"
    )
    size: str | None = Field(
        default=None, description="Requested image size, e.g. '1024x1024'"
    )
    format: ImageFormat = Field(
        default=ImageFormat.PNG, description="Output image format"
    )


class GenerateImageTool(Tool):
    name = "generate_image"
    description = "Generate an image from a text prompt."
    input_schema = GenerateImageInput
    output_schema = ImageDocument

    def __init__(
        self,
        generator: BaseImageGenerator,
        default_config: ImageGenConfig | None = None,
    ) -> None:
        self._generator = generator
        self._default_config = default_config

    async def run(self, **kwargs: Any) -> ImageDocument:
        validated = GenerateImageInput(**kwargs)
        return await safe_call(
            self._generator.generate(
                validated.prompt,
                config=self._default_config,
                size=validated.size,
                format=validated.format,
            ),
            "Image generation failed",
        )
