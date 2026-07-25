import pytest

from agent_platform.components.llm_classifier.config import ClassificationMode
from agent_platform.components.llm_classifier.strategies.prompt_templates import (
    build_classification_template,
)


class TestBuildClassificationTemplate:
    def test_zero_shot_single_label(self):
        template = build_classification_template(
            strategy=ClassificationMode.ZERO_SHOT, multi_label=False
        )
        assert "{labels}" in template
        assert "{unknown}" in template
        assert "record_classification" in template

    def test_zero_shot_multi_label(self):
        template = build_classification_template(
            strategy=ClassificationMode.ZERO_SHOT, multi_label=True
        )
        assert "zero, one, or multiple categories" in template
        assert "empty list" in template

    def test_few_shot_includes_examples_placeholder(self):
        template = build_classification_template(
            strategy=ClassificationMode.FEW_SHOT, multi_label=False
        )
        assert "{examples}" in template

    def test_text_classification_has_preamble(self):
        template = build_classification_template(
            strategy=ClassificationMode.TEXT_CLASSIFICATION, multi_label=False
        )
        assert "text-classification system" in template

    def test_sentiment_analysis_has_preamble(self):
        template = build_classification_template(
            strategy=ClassificationMode.SENTIMENT_ANALYSIS, multi_label=False
        )
        assert "sentiment-analysis system" in template

    def test_unsupported_strategy_raises(self):
        with pytest.raises(ValueError, match="Unsupported classification strategy"):
            build_classification_template(strategy="bogus", multi_label=False)
