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


def test_default_config_new_field_defaults():
    provider = MarkdownStructureChunkerProvider()
    cfg = provider._default_config()
    assert cfg.return_each_line is False
    assert cfg.custom_header_patterns is None


def test_chunk_forwards_return_each_line(mocker):
    provider = MarkdownStructureChunkerProvider()
    doc = TextDocument(text=MARKDOWN_DOC, format=DocumentFormat.MARKDOWN)
    config = MarkdownChunkerConfig(return_each_line=True)

    mock_splitter_cls = mocker.patch(
        "agent_platform.integrations.chunking.markdown.markdown.MarkdownHeaderTextSplitter"
    )
    mock_splitter_cls.return_value.split_text.return_value = []
    provider.chunk([doc], config)

    _, kwargs = mock_splitter_cls.call_args
    assert kwargs["return_each_line"] is True


def test_chunk_forwards_custom_header_patterns_when_set(mocker):
    provider = MarkdownStructureChunkerProvider()
    doc = TextDocument(text=MARKDOWN_DOC, format=DocumentFormat.MARKDOWN)
    config = MarkdownChunkerConfig(custom_header_patterns={"**": 1})

    mock_splitter_cls = mocker.patch(
        "agent_platform.integrations.chunking.markdown.markdown.MarkdownHeaderTextSplitter"
    )
    mock_splitter_cls.return_value.split_text.return_value = []
    provider.chunk([doc], config)

    _, kwargs = mock_splitter_cls.call_args
    assert kwargs["custom_header_patterns"] == {"**": 1}


def test_chunk_omits_custom_header_patterns_when_unset(mocker):
    provider = MarkdownStructureChunkerProvider()
    doc = TextDocument(text=MARKDOWN_DOC, format=DocumentFormat.MARKDOWN)

    mock_splitter_cls = mocker.patch(
        "agent_platform.integrations.chunking.markdown.markdown.MarkdownHeaderTextSplitter"
    )
    mock_splitter_cls.return_value.split_text.return_value = []
    provider.chunk([doc], None)

    _, kwargs = mock_splitter_cls.call_args
    assert "custom_header_patterns" not in kwargs


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
