from uuid import uuid4

import pytest

pytest.importorskip("langchain_text_splitters")

from agent_platform.integrations.chunking.config import (
    ChunkerConfig,
    RecursiveChunkerConfig,
)
from agent_platform.integrations.chunking.providers.recursive import (
    RecursiveChunkerProvider,
)
from agent_platform.integrations.chunking.langchain_base import LangChainChunker


def test_recursive_chunker_provider_defaults():
    provider = RecursiveChunkerProvider()
    assert provider.chunk_size == 512
    assert provider.chunk_overlap == 64
    assert provider.add_start_index is False


def test_recursive_chunker_provider_custom():
    provider = RecursiveChunkerProvider(
        chunk_size=256,
        chunk_overlap=32,
        add_start_index=True,
    )
    assert provider.chunk_size == 256
    assert provider.chunk_overlap == 32
    assert provider.add_start_index is True


def test_recursive_chunker_provider_is_langchain_chunker():
    provider = RecursiveChunkerProvider()
    assert isinstance(provider, LangChainChunker)


def test_splitter_applies_config():
    provider = RecursiveChunkerProvider(chunk_size=999, chunk_overlap=50)
    config = RecursiveChunkerConfig(chunk_size=128, chunk_overlap=16)
    splitter = provider._splitter(config)
    assert splitter._chunk_size == 128
    assert splitter._chunk_overlap == 16


def test_splitter_falls_back_to_instance_defaults():
    provider = RecursiveChunkerProvider(chunk_size=200, chunk_overlap=30)
    splitter = provider._splitter(None)
    assert splitter._chunk_size == 200
    assert splitter._chunk_overlap == 30


def test_splitter_always_adds_start_index():
    provider = RecursiveChunkerProvider()
    config = RecursiveChunkerConfig()
    splitter = provider._splitter(config)
    assert splitter._add_start_index is True
