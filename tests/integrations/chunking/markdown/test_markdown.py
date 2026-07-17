import pytest

pytest.importorskip("langchain_text_splitters")

from agent_platform.core.errors import ValidationError
from agent_platform.core.schemas.document import TextDocument
from agent_platform.core.schemas.enums import DocumentFormat
from agent_platform.integrations.chunking.markdown.config import MarkdownChunkerConfig
from agent_platform.integrations.chunking.markdown.markdown import (
    MarkdownStructureChunkerProvider,
)

MARKDOWN_DOC = (
    "# Title\n\n"
    "Intro paragraph text. "
    * 5
    + "\n\n## Section One\n\n"
    + "Section one body text. " * 5
    + "\n\n## Section Two\n\n"
    + "Section two body text. " * 5
)


def test_provider_defaults():
    provider = MarkdownStructureChunkerProvider()
    assert isinstance(provider, MarkdownStructureChunkerProvider)


def test_default_config_type():
    provider = MarkdownStructureChunkerProvider()
    cfg = provider._default_config()
    assert isinstance(cfg, MarkdownChunkerConfig)
    assert cfg.headers_to_split_on == [("#", "h1"), ("##", "h2"), ("###", "h3")]


def test_chunk_splits_by_headers_and_size():
    provider = MarkdownStructureChunkerProvider()
    doc = TextDocument(text=MARKDOWN_DOC, format=DocumentFormat.MARKDOWN)
    config = MarkdownChunkerConfig(chunk_size=80, chunk_overlap=0)

    chunks = provider.chunk([doc], config)

    assert len(chunks) > 1
    assert all(c.format == DocumentFormat.MARKDOWN for c in chunks)
    assert all(c.document_id == doc.id for c in chunks)
    assert any("h2" in c.metadata["headers"] for c in chunks)


def test_chunk_falls_back_to_default_config():
    provider = MarkdownStructureChunkerProvider()
    doc = TextDocument(text=MARKDOWN_DOC, format=DocumentFormat.MARKDOWN)

    chunks = provider.chunk([doc], None)

    assert len(chunks) >= 1


def test_chunk_empty_document_returns_no_chunks():
    provider = MarkdownStructureChunkerProvider()
    doc = TextDocument(text="", format=DocumentFormat.MARKDOWN)

    chunks = provider.chunk([doc], None)

    assert chunks == []


def test_chunk_accepts_unknown_format():
    provider = MarkdownStructureChunkerProvider()
    doc = TextDocument(text=MARKDOWN_DOC, format=DocumentFormat.UNKNOWN)

    chunks = provider.chunk([doc], None)

    assert len(chunks) >= 1


def test_chunk_rejects_mismatched_format():
    provider = MarkdownStructureChunkerProvider()
    doc = TextDocument(text=MARKDOWN_DOC, format=DocumentFormat.HTML)

    with pytest.raises(ValidationError):
        provider.chunk([doc], None)
