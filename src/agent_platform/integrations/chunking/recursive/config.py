from __future__ import annotations
from pydantic import BaseModel, ConfigDict, Field
from agent_platform.core.interfaces.chunking.config import ChunkerConfig


class RecursiveChunkerConfig(ChunkerConfig):
    model_config = ConfigDict(populate_by_name=True, frozen=True)
    separators: list[str] | None = Field(default=["\n\n", "\n", " ", ""])
    add_start_index: bool = False
