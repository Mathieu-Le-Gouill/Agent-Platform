import pytest

pytest.importorskip("langchain_text_splitters")

from agent_platform.core.interfaces.chunking.base import BaseChunker
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.document import TextDocument
from agent_platform.core.schemas.enums import DocumentFormat
from agent_platform.integrations.chunking.langchain_base import (
    LangChainChunker,
    _doc_to_lc,
)
from agent_platform.integrations.chunking.recursive.config import RecursiveChunkerConfig
from agent_platform.integrations.chunking.recursive.recursive import (
    RecursiveChunkerProvider,
)


def test_recursive_chunker_provider_defaults():
    provider = RecursiveChunkerProvider()
    assert isinstance(provider, RecursiveChunkerProvider)


def test_recursive_chunker_provider_is_langchain_chunker():
    provider = RecursiveChunkerProvider()
    assert isinstance(provider, LangChainChunker)


# Replaced brittle private-attribute assertions with behavioral tests


def test_splitter_applies_config():
    provider = RecursiveChunkerProvider()
    config = RecursiveChunkerConfig(
        chunk_size=128, chunk_overlap=16, add_start_index=True
    )
    doc = TextDocument(
        text="Hello world. This is a test." * 10,
        format=DocumentFormat.TXT,
    )
    splitter = provider._splitter(config)
    result = splitter.split_documents([_doc_to_lc(doc)])

    assert len(result) > 0
    assert all(len(c.page_content) <= 128 for c in result)


def test_splitter_falls_back_to_defaults():
    provider = RecursiveChunkerProvider()
    doc = TextDocument(
        text="The quick brown fox jumps over the lazy dog. " * 20,
        format=DocumentFormat.TXT,
    )
    splitter = provider._splitter(RecursiveChunkerConfig())
    result = splitter.split_documents([_doc_to_lc(doc)])

    assert len(result) > 0


def test_splitter_applies_add_start_index():
    provider = RecursiveChunkerProvider()
    config = RecursiveChunkerConfig(
        chunk_size=50, chunk_overlap=0, add_start_index=True
    )
    doc = TextDocument(text="Hello world. " * 5, format=DocumentFormat.TXT)
    splitter = provider._splitter(config)

    lc_docs = splitter.split_documents([_doc_to_lc(doc)])
    start_indices = [lc.metadata.get("start_index") for lc in lc_docs]

    assert any(i is not None for i in start_indices)


def test_splitter_without_add_start_index():
    provider = RecursiveChunkerProvider()
    config = RecursiveChunkerConfig(
        chunk_size=50, chunk_overlap=0, add_start_index=False
    )
    doc = TextDocument(text="Hello world. " * 5, format=DocumentFormat.TXT)
    splitter = provider._splitter(config)

    lc_docs = splitter.split_documents([_doc_to_lc(doc)])
    start_indices = [lc.metadata.get("start_index") for lc in lc_docs]

    assert all(i is None for i in start_indices)


def test_splitter_forwards_new_fields_to_constructor(mocker):
    provider = RecursiveChunkerProvider()
    config = RecursiveChunkerConfig(
        keep_separator="start",
        is_separator_regex=True,
        strip_whitespace=False,
    )

    mock_splitter_cls = mocker.patch(
        "agent_platform.integrations.chunking.recursive.recursive.RecursiveCharacterTextSplitter"
    )
    provider._splitter(config)

    _, kwargs = mock_splitter_cls.call_args
    assert kwargs["keep_separator"] == "start"
    assert kwargs["is_separator_regex"] is True
    assert kwargs["strip_whitespace"] is False


def test_splitter_applies_strip_whitespace_false():
    provider = RecursiveChunkerProvider()
    config = RecursiveChunkerConfig(
        chunk_size=8, chunk_overlap=0, strip_whitespace=False, separators=[" "]
    )
    doc = TextDocument(text="one two three four five six", format=DocumentFormat.TXT)
    splitter = provider._splitter(config)

    result = splitter.split_documents([_doc_to_lc(doc)])

    assert any(c.page_content != c.page_content.strip() for c in result)


def test_splitter_different_separators():
    provider = RecursiveChunkerProvider()
    config = RecursiveChunkerConfig(chunk_size=10, chunk_overlap=0, separators=[" "])
    doc = TextDocument(text="one two three four five", format=DocumentFormat.TXT)
    splitter = provider._splitter(config)
    result = splitter.split_documents([_doc_to_lc(doc)])

    assert len(result) >= 2


def test_default_config_type():
    provider = RecursiveChunkerProvider()
    cfg = provider._default_config()
    assert isinstance(cfg, RecursiveChunkerConfig)
    assert cfg.chunk_size == 512
    assert cfg.chunk_overlap == 64


def test_base_chunker_subclass_can_store_credentials():
    marker = object()

    class _ConcreteChunker(
        BaseChunker[TextDocument, TextChunk, RecursiveChunkerConfig]
    ):
        def __init__(self, credentials):
            self._credentials = credentials

        def chunk(self, documents, config):
            return []

    chunker = _ConcreteChunker(marker)
    assert chunker._credentials is marker
