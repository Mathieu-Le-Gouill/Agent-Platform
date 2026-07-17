from __future__ import annotations

from agent_platform.core.interfaces.chunking.config import ChunkerConfig


class LatexChunkerConfig(ChunkerConfig):
    add_start_index: bool = False
