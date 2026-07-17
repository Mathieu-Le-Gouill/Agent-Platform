from __future__ import annotations

from agent_platform.core.interfaces.chunking.config import ChunkerConfig


class PDFChunkerConfig(ChunkerConfig):
    combine_text_under_n_chars: int = 0
    new_after_n_chars: int | None = None
    multipage_sections: bool = True
