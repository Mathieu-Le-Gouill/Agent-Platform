from __future__ import annotations

from pydantic import Field

from agent_platform.core.interfaces.chunking.config import ChunkerConfig


class HTMLChunkerConfig(ChunkerConfig):
    # List of (HTML tag, semantic name) pairs used to split the document at
    # matching header tags.
    headers_to_split_on: list[tuple[str, str]] = Field(
        default_factory=lambda: [
            ("h1", "h1"),
            ("h2", "h2"),
            ("h3", "h3"),
        ]
    )
    # If True, emits one chunk per individual HTML element instead of
    # grouping elements under the same header into a single chunk.
    return_each_element: bool = False


# sources: https://reference.langchain.com/python/langchain-text-splitters/html/HTMLHeaderTextSplitter
