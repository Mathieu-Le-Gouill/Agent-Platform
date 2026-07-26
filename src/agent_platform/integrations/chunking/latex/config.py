from __future__ import annotations

from typing import Literal

from agent_platform.core.interfaces.chunking.config import ChunkerConfig


class LatexChunkerConfig(ChunkerConfig):
    # If True, adds a "start_index" metadata field with the chunk's offset in
    # the original document.
    add_start_index: bool = False
    # Note: no `separators` field here — from_language(Language.LATEX, ...) derives
    # separators internally from the LaTeX grammar; a custom value would be ignored.
    keep_separator: bool | Literal["start", "end"] = True
    # Exposed for parity with `recursive`, but not forwarded to the splitter:
    # from_language() always passes is_separator_regex=True internally.
    is_separator_regex: bool = False
    # If True, strips leading/trailing whitespace from each emitted chunk.
    strip_whitespace: bool = True


"""
sources: https://reference.langchain.com/python/langchain-text-splitters/base/TextSplitter
         https://reference.langchain.com/python/langchain-text-splitters/character/RecursiveCharacterTextSplitter
"""
