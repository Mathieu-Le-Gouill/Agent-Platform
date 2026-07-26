from __future__ import annotations

from agent_platform.core.interfaces.chunking.config import ChunkerConfig


class PDFChunkerConfig(ChunkerConfig):
    # 0 disables combining short elements entirely; the library's own default
    # is `max_characters` (aggressive combining). Deliberate deviation.
    combine_text_under_n_chars: int = 0
    # Soft cap: a chunk is closed and a new one started once it exceeds this
    # many characters, even if `chunk_size`/`max_characters` isn't yet reached.
    new_after_n_chars: int | None = None
    # With this True, merged chunks spanning pages typically keep only the
    # first element's page_number — known limitation, not fixed here.
    multipage_sections: bool = True
    # chunk_by_title's `overlap` (base `chunk_overlap`) only applies when an
    # individual element must itself be split for exceeding `max_characters`;
    # it does NOT add overlap between normally-combined chunks unless this is
    # also True. Defaults to True so `chunk_overlap` behaves consistently
    # with `recursive`/`latex`'s inter-chunk overlap semantics.
    overlap_all: bool = True
    # If True, keeps the original (pre-combination) elements accessible in
    # each chunk's metadata alongside the combined chunk text.
    include_orig_elements: bool = False
    # Token-based chunking alternative to max_characters; mutually exclusive
    # with it. When set, `chunk_size` (max_characters) is omitted from the
    # chunk_by_title call and `tokenizer` must also be set.
    max_tokens: int | None = None
    # Encoding name (e.g. "cl100k_base") or model name (e.g. "gpt-4") used to
    # count tokens when `max_tokens` is set; required by the library in that case.
    tokenizer: str | None = None
    # If True, tables are left intact (not split by `chunk_size`) instead of
    # being chunked like other elements.
    skip_table_chunking: bool = False
    # If True, repeats the table's header row in each chunk produced when a
    # table is split across multiple chunks.
    repeat_table_headers: bool = False
    # If True, keeps each table as its own chunk, never combined with
    # surrounding non-table elements.
    isolate_table: bool = False


# sources: https://docs.unstructured.io/open-source/core-functionality/chunking#by-title-chunking-strategy
