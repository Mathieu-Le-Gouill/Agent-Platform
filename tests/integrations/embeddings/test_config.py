import pytest

from agent_platform.core.interfaces.embeddings.config import EmbeddingConfig
from agent_platform.integrations.embeddings.huggingface.config import (
    HuggingFaceEmbeddingConfig,
    HuggingFaceEmbeddingMode,
)
from agent_platform.integrations.embeddings.mistral.config import MistralEmbeddingConfig
from agent_platform.integrations.embeddings.ollama.config import OllamaEmbeddingConfig
from agent_platform.integrations.embeddings.openai.config import OpenAIEmbeddingConfig


class TestEmbeddingConfig:
    def test_defaults(self):
        cfg = EmbeddingConfig()
        assert cfg.batch_size == 32
        assert cfg.dimensions is None
        assert cfg.timeout is None

    @pytest.mark.parametrize(
        ("batch_size", "dimensions", "timeout"),
        [
            (64, 768, 30.0),
            (1, None, None),
            (128, 512, 60.0),
        ],
    )
    def test_construction(self, batch_size, dimensions, timeout):
        cfg = EmbeddingConfig(
            batch_size=batch_size,
            dimensions=dimensions,
            timeout=timeout,
        )
        assert cfg.batch_size == batch_size
        assert cfg.dimensions == dimensions
        assert cfg.timeout == timeout


class TestOpenAIEmbeddingConfig:
    def test_defaults(self):
        cfg = OpenAIEmbeddingConfig()
        assert cfg.model == "text-embedding-ada-002"
        assert cfg.batch_size == 32
        assert cfg.max_retries is None

    def test_construction(self):
        cfg = OpenAIEmbeddingConfig(model="text-embedding-3-small", dimensions=256)
        assert cfg.model == "text-embedding-3-small"
        assert cfg.dimensions == 256

    def test_model_kwargs(self):
        cfg = OpenAIEmbeddingConfig(model_kwargs={"foo": "bar"})
        assert cfg.model_kwargs == {"foo": "bar"}

    def test_dimensions_omitted_by_default(self):
        cfg = OpenAIEmbeddingConfig()
        assert cfg.dimensions is None


class TestHuggingFaceEmbeddingConfig:
    def test_defaults(self):
        cfg = HuggingFaceEmbeddingConfig()
        assert cfg.model == "sentence-transformers/all-MiniLM-L6-v2"
        assert cfg.model_kwargs is None
        assert cfg.encode_kwargs is None
        assert cfg.mode == HuggingFaceEmbeddingMode.LOCAL

    def test_construction(self):
        cfg = HuggingFaceEmbeddingConfig(
            model_kwargs={"device": "cpu"},
            encode_kwargs={"show_progress_bar": True},
        )
        assert cfg.model_kwargs == {"device": "cpu"}
        assert cfg.encode_kwargs == {"show_progress_bar": True}

    def test_hosted_mode(self):
        cfg = HuggingFaceEmbeddingConfig(mode=HuggingFaceEmbeddingMode.HOSTED)
        assert cfg.mode == HuggingFaceEmbeddingMode.HOSTED

    def test_mode_values(self):
        assert HuggingFaceEmbeddingMode.LOCAL.value == "local"
        assert HuggingFaceEmbeddingMode.HOSTED.value == "hosted"


class TestMistralEmbeddingConfig:
    def test_defaults(self):
        cfg = MistralEmbeddingConfig()
        assert cfg.timeout is None
        assert cfg.endpoint == "https://api.mistral.ai/v1/"
        assert cfg.wait_time is None
        assert cfg.max_concurrent_requests is None

    def test_construction(self):
        cfg = MistralEmbeddingConfig(timeout=60)
        assert cfg.timeout == 60

    def test_endpoint_override(self):
        cfg = MistralEmbeddingConfig(endpoint="http://localhost:8080")
        assert cfg.endpoint == "http://localhost:8080"

    def test_wait_time(self):
        cfg = MistralEmbeddingConfig(wait_time=5)
        assert cfg.wait_time == 5


class TestOllamaEmbeddingConfig:
    def test_defaults(self):
        cfg = OllamaEmbeddingConfig()
        assert cfg.model == "nomic-embed-text"
        assert cfg.temperature is None
        assert cfg.top_p is None
        assert cfg.top_k is None

    def test_construction(self):
        cfg = OllamaEmbeddingConfig(model="llama-embed")
        assert cfg.model == "llama-embed"
