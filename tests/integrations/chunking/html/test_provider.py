import pytest

pytest.importorskip("langchain_text_splitters")
pytest.importorskip("bs4")

from agent_platform.core.errors import ValidationError
from agent_platform.core.schemas.document import TextDocument
from agent_platform.core.schemas.enums import DocumentFormat
from agent_platform.integrations.chunking.html.config import HTMLChunkerConfig
from agent_platform.integrations.chunking.html.provider import (
    HTMLStructureChunkerProvider,
)

HTML_DOC = (
    "<html><body>"
    "<h1>Title</h1><p>" + ("Intro paragraph text. " * 5) + "</p>"
    "<h2>Section One</h2><p>" + ("Section one body text. " * 5) + "</p>"
    "<h2>Section Two</h2><p>" + ("Section two body text. " * 5) + "</p>"
    "</body></html>"
)


def test_provider_defaults():
    provider = HTMLStructureChunkerProvider()
    assert isinstance(provider, HTMLStructureChunkerProvider)


def test_default_config_type():
    provider = HTMLStructureChunkerProvider()
    cfg = provider._default_config()
    assert isinstance(cfg, HTMLChunkerConfig)
    assert cfg.headers_to_split_on == [("h1", "h1"), ("h2", "h2"), ("h3", "h3")]


def test_default_config_return_each_element_default():
    provider = HTMLStructureChunkerProvider()
    cfg = provider._default_config()
    assert cfg.return_each_element is False


def test_chunk_forwards_return_each_element(mocker):
    provider = HTMLStructureChunkerProvider()
    doc = TextDocument(text=HTML_DOC, format=DocumentFormat.HTML)
    config = HTMLChunkerConfig(return_each_element=True)

    mock_splitter_cls = mocker.patch(
        "agent_platform.integrations.chunking.html.provider.HTMLHeaderTextSplitter"
    )
    mock_splitter_cls.return_value.split_text.return_value = []
    provider.chunk([doc], config)

    _, kwargs = mock_splitter_cls.call_args
    assert kwargs["return_each_element"] is True


def test_chunk_splits_by_headers_and_size():
    provider = HTMLStructureChunkerProvider()
    doc = TextDocument(text=HTML_DOC, format=DocumentFormat.HTML)
    config = HTMLChunkerConfig(chunk_size=80, chunk_overlap=0)

    chunks = provider.chunk([doc], config)

    assert len(chunks) > 1
    assert all(c.format == DocumentFormat.HTML for c in chunks)
    assert all(c.document_id == doc.id for c in chunks)
    assert any("h2" in c.metadata["headers"] for c in chunks)


def test_chunk_falls_back_to_default_config():
    provider = HTMLStructureChunkerProvider()
    doc = TextDocument(text=HTML_DOC, format=DocumentFormat.HTML)

    chunks = provider.chunk([doc], None)

    assert len(chunks) >= 1


def test_chunk_accepts_unknown_format():
    provider = HTMLStructureChunkerProvider()
    doc = TextDocument(text=HTML_DOC, format=DocumentFormat.UNKNOWN)

    chunks = provider.chunk([doc], None)

    assert len(chunks) >= 1


def test_chunk_rejects_mismatched_format():
    provider = HTMLStructureChunkerProvider()
    doc = TextDocument(text=HTML_DOC, format=DocumentFormat.MARKDOWN)

    with pytest.raises(ValidationError):
        provider.chunk([doc], None)
