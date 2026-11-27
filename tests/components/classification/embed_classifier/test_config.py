import pytest
from pydantic import ValidationError

from agent_platform.components.classification.embed_classifier.config import (
    EmbeddingClassifierConfig,
)
from agent_platform.core.schemas.enums import SimilarityMetric


class TestEmbeddingClassifierConfig:
    def test_defaults(self):
        config = EmbeddingClassifierConfig()
        assert config.top_k == 5
        assert config.similarity_threshold is None
        assert config.multi_label is False
        assert config.unknown_label == "unknown"
        assert config.similarity_metric == SimilarityMetric.COSINE

    def test_construction(self):
        config = EmbeddingClassifierConfig(top_k=3, multi_label=True)
        assert config.top_k == 3
        assert config.multi_label is True

    def test_rejects_top_k_below_one(self):
        with pytest.raises(ValidationError, match="top_k"):
            EmbeddingClassifierConfig(top_k=0)
