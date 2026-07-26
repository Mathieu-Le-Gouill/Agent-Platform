from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from agent_platform.core.interfaces.image_generation.config import ImageGenConfig

# Quality vocabulary differs per model
_QUALITY_BY_MODEL: dict[str, tuple[str, ...]] = {
    "dall-e-3": ("standard", "hd"),
    "gpt-image-1": ("high", "medium", "low", "auto"),
}


class DalleConfig(ImageGenConfig):
    # Model id: "dall-e-2", "dall-e-3", or "gpt-image-1".
    model: str = "dall-e-3"
    # Image quality: dall-e-3 accepts "standard"|"hd", gpt-image-1 accepts "high"|"medium"|"low"|"auto",
    # dall-e-2 does not support this param at all.
    quality: str = "standard"
    # Style preset, dall-e-3 only ("vivid" = hyper-real/dramatic, "natural" = more literal).
    style: Literal["vivid", "natural"] | None = None

    @model_validator(mode="after")
    def _validate_quality(self) -> DalleConfig:
        allowed = _QUALITY_BY_MODEL.get(self.model)
        if allowed is not None and self.quality not in allowed:
            raise ValueError(
                f"Invalid quality '{self.quality}' for {self.model}. Valid: {allowed}"
            )
        return self


# sources: https://developers.openai.com/api/reference/resources/images
