from __future__ import annotations

from typing import Literal

from pydantic import Field

from agent_platform.core.interfaces.chunking.config import ChunkerConfig


class RecursiveChunkerConfig(ChunkerConfig):
    # Ordered list of strings tried, in order, to split text into chunks; the
    # splitter recurses into the next separator when a chunk is still too large.
    separators: list[str] | None = Field(default=["\n\n", "\n", " ", ""])
    # If True, adds a "start_index" metadata field with the chunk's offset in
    # the original document.
    add_start_index: bool = False
    # Controls whether/where the separator is kept: False drops it, True (or
    # "start"/"end") keeps it attached to the start or end of the chunk.
    keep_separator: bool | Literal["start", "end"] = True
    # If True, treats each entry in `separators` as a regex pattern instead
    # of a literal string.
    is_separator_regex: bool = False
    # If True, strips leading/trailing whitespace from each chunk.
    strip_whitespace: bool = True


"""
sources: https://reference.langchain.com/python/langchain-text-splitters/character/RecursiveCharacterTextSplitter
         https://reference.langchain.com/python/langchain-text-splitters/base/TextSplitter
"""
