from __future__ import annotations
from pydantic import Field
from agent_platform.core.interfaces.chunking.config import ChunkerConfig


class RecursiveChunkerConfig(ChunkerConfig):
    separators: list[str] | None = Field(default=["\n\n", "\n", " ", ""])
    add_start_index: bool = False
