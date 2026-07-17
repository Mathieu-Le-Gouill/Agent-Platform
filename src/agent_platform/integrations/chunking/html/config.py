from __future__ import annotations
from pydantic import Field

from agent_platform.core.interfaces.chunking.config import ChunkerConfig


class HTMLChunkerConfig(ChunkerConfig):
    headers_to_split_on: list[tuple[str, str]] = Field(
        default_factory=lambda: [
            ("h1", "h1"),
            ("h2", "h2"),
            ("h3", "h3"),
        ]
    )
