from __future__ import annotations
from pydantic import Field

from agent_platform.core.interfaces.chunking.config import ChunkerConfig


class MarkdownChunkerConfig(ChunkerConfig):
    # List of (header marker, semantic name) pairs, e.g. ("#", "h1"), used to
    # split the document at matching Markdown headers.
    headers_to_split_on: list[tuple[str, str]] = Field(
        default_factory=lambda: [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]
    )
    # Deliberate override: the library defaults to True (strip headers from
    # chunk content); this platform keeps headers in the text by default so
    # downstream consumers retain heading context inline, not an oversight.
    strip_headers: bool = False
    # If True, splits output into one chunk per line instead of grouping
    # consecutive lines under the same header into a single chunk.
    return_each_line: bool = False
    # Optional mapping of custom header marker -> header level, merged with
    # `headers_to_split_on` to recognize non-standard header syntaxes.
    custom_header_patterns: dict[str, int] | None = None


# sources: https://python.langchain.com/api_reference/text_splitters/markdown/langchain_text_splitters.markdown.MarkdownHeaderTextSplitter.html
