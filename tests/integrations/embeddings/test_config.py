from dataclasses import FrozenInstanceError

import pytest

from agent_platform.integrations.embeddings.config import (
    EmbeddingConfig,
    OpenAIEmbeddingConfig,
    SentenceTransformerConfig,
    MistralEmbeddingConfig,
    OllamaEmbeddingConfig,
)


class TestEmbeddingConfig:
    def test_defaults(self):
        cfg = EmbeddingConfig()
        assert cfg.batch_size == 32
        assert cfg.normalize is True
        assert cfg.truncate is True
        assert cfg.dimensions is None
        assert cfg.device is None
        assert cfg.show_progress is False

    def test_construction(self):
        cfg = EmbeddingConfig(
            batch_size=64,
            normalize=False,
            truncate=False,
            dimensions=768,
            device="cuda",
            show_progress=True,
        )
        assert cfg.batch_size == 64
        assert cfg.normalize is False
        assert cfg.truncate is False
        assert cfg.dimensions == 768
        assert cfg.device == "cuda"
        assert cfg.show_progress is True

    def test_frozen(self):
        cfg = EmbeddingConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.batch_size = 16


class TestOpenAIEmbeddingConfig:
    def test_defaults(self):
        cfg = OpenAIEmbeddingConfig()
        assert cfg.organization is None
        assert cfg.timeout is None
        assert cfg.batch_size == 32

    def test_construction(self):
        cfg = OpenAIEmbeddingConfig(
            organization="org-123",
            timeout=30.0,
            dimensions=256,
        )
        assert cfg.organization == "org-123"
        assert cfg.timeout == 30.0
        assert cfg.dimensions == 256

    def test_frozen(self):
        cfg = OpenAIEmbeddingConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.organization = "org-other"


class TestSentenceTransformerConfig:
    def test_defaults(self):
        cfg = SentenceTransformerConfig()
        assert cfg.model_kwargs is None
        assert cfg.encode_kwargs is None

    def test_construction(self):
        cfg = SentenceTransformerConfig(
            model_kwargs={"device": "cpu"},
            encode_kwargs={"show_progress_bar": True},
        )
        assert cfg.model_kwargs == {"device": "cpu"}
        assert cfg.encode_kwargs == {"show_progress_bar": True}

    def test_frozen(self):
        cfg = SentenceTransformerConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.model_kwargs = {}


class TestMistralEmbeddingConfig:
    def test_defaults(self):
        cfg = MistralEmbeddingConfig()
        assert cfg.timeout is None

    def test_construction(self):
        cfg = MistralEmbeddingConfig(timeout=60)
        assert cfg.timeout == 60

    def test_frozen(self):
        cfg = MistralEmbeddingConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.timeout = 30


class TestOllamaEmbeddingConfig:
    def test_defaults(self):
        cfg = OllamaEmbeddingConfig()
        assert cfg.base_url == "http://localhost:11434"

    def test_construction(self):
        cfg = OllamaEmbeddingConfig(base_url="http://10.0.0.1:11434")
        assert cfg.base_url == "http://10.0.0.1:11434"

    def test_frozen(self):
        cfg = OllamaEmbeddingConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.base_url = "http://other:11434"
