import pytest

from agent_platform.components.llm_classifier.config import ClassificationMode
from agent_platform.components.llm_classifier.strategies.registry import get_strategy
from agent_platform.components.llm_classifier.strategies.sentiment import (
    SentimentStrategy,
)
from agent_platform.components.llm_classifier.strategies.text_classification import (
    TextClassificationStrategy,
)
from agent_platform.components.llm_classifier.strategies.zero_shot import (
    ZeroShotStrategy,
)


class TestGetStrategy:
    def test_zero_shot_registered(self):
        assert isinstance(get_strategy(ClassificationMode.ZERO_SHOT), ZeroShotStrategy)

    def test_text_classification_registered(self):
        assert isinstance(
            get_strategy(ClassificationMode.TEXT_CLASSIFICATION),
            TextClassificationStrategy,
        )

    def test_sentiment_registered(self):
        assert isinstance(
            get_strategy(ClassificationMode.SENTIMENT_ANALYSIS), SentimentStrategy
        )

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="No strategy registered"):
            get_strategy("bogus_mode")
