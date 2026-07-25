from agent_platform.components.llm_classifier.config import LLMClassifierConfig
from agent_platform.components.llm_classifier.strategies.zero_shot import (
    ZeroShotStrategy,
)


class TestZeroShotStrategy:
    def test_single_label_includes_labels_and_unknown(self):
        strategy = ZeroShotStrategy()
        prompt = strategy.build_system_prompt(
            LLMClassifierConfig(unknown_label="none"),
            ["cat", "dog"],
            multi_label=False,
        )
        assert "cat, dog" in prompt
        assert "none" in prompt

    def test_multi_label_omits_unknown_placeholder(self):
        strategy = ZeroShotStrategy()
        prompt = strategy.build_system_prompt(
            LLMClassifierConfig(), ["cat", "dog"], multi_label=True
        )
        assert "cat, dog" in prompt
