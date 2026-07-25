from agent_platform.components.embed_classifier.config import EmbeddingClassifierConfig
from agent_platform.core.schemas.enums import SimilarityMetric


class TestEmbeddingClassifierConfig:
    def test_defaults(self):
        config = EmbeddingClassifierConfig()
        assert config.top_k == 5
        assert config.similarity_threshold is None
        assert config.multi_label is False
        assert config.unknown_label == "unknown"
        assert config.similarity_metric == SimilarityMetric.COSINE
        assert config.llm_rerank is False
        assert config.llm_classifier_config is None

    def test_construction(self):
        config = EmbeddingClassifierConfig(top_k=3, multi_label=True, llm_rerank=True)
        assert config.top_k == 3
        assert config.multi_label is True
        assert config.llm_rerank is True
