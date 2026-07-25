from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ProviderConfig"]


class ProviderConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, frozen=True)

    extra_params: dict[str, Any] = Field(default_factory=dict)
