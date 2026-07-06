from uuid import uuid4

import pytest

from agent_platform.integrations.chunking.config import (
    ChunkerConfig,
    RecursiveChunkerConfig,
)


def test_chunker_config_defaults():
    cfg = ChunkerConfig()
    assert cfg.chunk_size == 512
    assert cfg.chunk_overlap == 64


def test_chunker_config_custom():
    cfg = ChunkerConfig(chunk_size=1024, chunk_overlap=128)
    assert cfg.chunk_size == 1024
    assert cfg.chunk_overlap == 128


def test_chunker_config_is_frozen():
    cfg = ChunkerConfig()
    with pytest.raises(AttributeError):
        cfg.chunk_size = 256


def test_recursive_chunker_config_inherits_defaults():
    cfg = RecursiveChunkerConfig()
    assert cfg.chunk_size == 512
    assert cfg.chunk_overlap == 64
    assert cfg.separators == ("\n\n", "\n", " ", "")


def test_recursive_chunker_config_custom_separators():
    cfg = RecursiveChunkerConfig(
        chunk_size=256,
        chunk_overlap=32,
        separators=("\n\n", "\n"),
    )
    assert cfg.chunk_size == 256
    assert cfg.chunk_overlap == 32
    assert cfg.separators == ("\n\n", "\n")


def test_recursive_chunker_config_is_frozen():
    cfg = RecursiveChunkerConfig()
    with pytest.raises(AttributeError):
        cfg.separators = ("\n\n",)


def test_chunker_config_is_dataclass():
    import dataclasses

    assert dataclasses.is_dataclass(ChunkerConfig)
    assert dataclasses.is_dataclass(RecursiveChunkerConfig)


def test_recursive_chunker_config_is_subclass():
    assert issubclass(RecursiveChunkerConfig, ChunkerConfig)
