from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ProviderConfig", "ModelConfig"]


class ProviderConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, frozen=True)

    extra_params: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(ProviderConfig):
    # Identifier of the model/deployment the provider calls; provider-specific
    # meaning, see each provider's config for its default override.
    model: str = ""
