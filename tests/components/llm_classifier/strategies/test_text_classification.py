from agent_platform.components.llm_classifier.config import LLMClassifierConfig
from agent_platform.components.llm_classifier.strategies.text_classification import (
    TextClassificationStrategy,
)


class TestTextClassificationStrategy:
    def test_single_label_prompt(self):
        strategy = TextClassificationStrategy()
        prompt = strategy.build_system_prompt(
            LLMClassifierConfig(unknown_label="none"),
            ["sports", "politics"],
            multi_label=False,
        )
        assert "sports, politics" in prompt
        assert "none" in prompt
        assert "text-classification system" in prompt

    def test_multi_label_prompt(self):
        strategy = TextClassificationStrategy()
        prompt = strategy.build_system_prompt(
            LLMClassifierConfig(), ["sports", "politics"], multi_label=True
        )
        assert "sports, politics" in prompt
