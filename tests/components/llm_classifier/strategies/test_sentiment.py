from agent_platform.components.llm_classifier.config import LLMClassifierConfig
from agent_platform.components.llm_classifier.strategies.sentiment import (
    SentimentStrategy,
)


class TestSentimentStrategy:
    def test_single_label_prompt(self):
        strategy = SentimentStrategy()
        prompt = strategy.build_system_prompt(
            LLMClassifierConfig(unknown_label="neutral"),
            ["positive", "negative"],
            multi_label=False,
        )
        assert "positive, negative" in prompt
        assert "neutral" in prompt
        assert "sentiment-analysis system" in prompt

    def test_multi_label_prompt(self):
        strategy = SentimentStrategy()
        prompt = strategy.build_system_prompt(
            LLMClassifierConfig(), ["positive", "negative"], multi_label=True
        )
        assert "positive, negative" in prompt
