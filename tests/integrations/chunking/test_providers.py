import pytest

pytest.importorskip("langchain_text_splitters")

from agent_platform.integrations.chunking.config import (
    RecursiveChunkerConfig,
)
from agent_platform.integrations.chunking.providers.recursive import (
    RecursiveChunkerProvider,
)
from agent_platform.integrations.chunking.langchain_base import (
    LangChainChunker,
    _doc_to_lc,
)
from agent_platform.integrations.chunking.base import BaseChunker
from agent_platform.integrations.credentials import NoCredentials
from agent_platform.models.chunk import TextChunk
from agent_platform.models.document import TextDocument
from agent_platform.models.enums import DocumentFormat


def test_recursive_chunker_provider_defaults():
    provider = RecursiveChunkerProvider()
    assert isinstance(provider._credentials, NoCredentials)


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


def test_splitter_different_separators():
    provider = RecursiveChunkerProvider()
    config = RecursiveChunkerConfig(
        chunk_size=10, chunk_overlap=0, separators=[" "]
    )
    doc = TextDocument(
        text="one two three four five", format=DocumentFormat.TXT
    )
    splitter = provider._splitter(config)
    result = splitter.split_documents([_doc_to_lc(doc)])

    assert len(result) >= 2


def test_default_config_type():
    provider = RecursiveChunkerProvider()
    cfg = provider._default_config()
    assert isinstance(cfg, RecursiveChunkerConfig)
    assert cfg.chunk_size == 512
    assert cfg.chunk_overlap == 64


def test_base_chunker_stores_credentials():
    class _ConcreteChunker(BaseChunker[NoCredentials, TextDocument, TextChunk, RecursiveChunkerConfig]):
        def chunk(self, documents, config):
            return []

    creds = NoCredentials()
    chunker = _ConcreteChunker(creds)
    assert chunker._credentials is creds
