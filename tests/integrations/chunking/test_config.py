import pytest

from agent_platform.core.interfaces.chunking.config import ChunkerConfig
from agent_platform.integrations.chunking.recursive.config import RecursiveChunkerConfig


def test_chunker_config_defaults():
    cfg = ChunkerConfig()
    assert cfg.chunk_size == 512
    assert cfg.chunk_overlap == 64


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (1024, 128),
        (256, 32),
        (1, 0),
    ],
)
def test_chunker_config_custom(chunk_size, chunk_overlap):
    cfg = ChunkerConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    assert cfg.chunk_size == chunk_size
    assert cfg.chunk_overlap == chunk_overlap


def test_recursive_chunker_config_inherits_defaults():
    cfg = RecursiveChunkerConfig()
    assert cfg.chunk_size == 512
    assert cfg.chunk_overlap == 64
    assert cfg.separators == ["\n\n", "\n", " ", ""]


def test_recursive_chunker_config_custom_separators():
    cfg = RecursiveChunkerConfig(
        chunk_size=256,
        chunk_overlap=32,
        separators=["\n\n", "\n"],
    )
    assert cfg.chunk_size == 256
    assert cfg.chunk_overlap == 32
    assert cfg.separators == ["\n\n", "\n"]


def test_recursive_chunker_config_add_start_index_default():
    cfg = RecursiveChunkerConfig()
    assert cfg.add_start_index is False


def test_recursive_chunker_config_add_start_index_true():
    cfg = RecursiveChunkerConfig(add_start_index=True)
    assert cfg.add_start_index is True


def test_recursive_chunker_config_is_subclass():
    assert issubclass(RecursiveChunkerConfig, ChunkerConfig)


def test_recursive_chunker_config_pydantic():
    from pydantic import BaseModel

    assert isinstance(ChunkerConfig(), BaseModel)
