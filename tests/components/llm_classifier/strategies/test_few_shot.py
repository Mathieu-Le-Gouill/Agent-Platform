import pytest

from agent_platform.components.llm_classifier.config import (
    ClassificationExample,
    LLMClassifierConfig,
)
from agent_platform.components.llm_classifier.strategies.few_shot import (
    FewShotStrategy,
)


class TestFewShotStrategy:
    def test_no_examples_raises(self):
        strategy = FewShotStrategy()
        with pytest.raises(ValueError, match="requires config.examples"):
            strategy.build_system_prompt(
                LLMClassifierConfig(), ["cat", "dog"], multi_label=False
            )

    def test_single_label_includes_examples(self):
        strategy = FewShotStrategy()
        config = LLMClassifierConfig(
            examples=[ClassificationExample(text="meow", label="cat")]
        )
        prompt = strategy.build_system_prompt(config, ["cat", "dog"], multi_label=False)
        assert "meow" in prompt
        assert "Label: cat" in prompt

    def test_multi_label_includes_unknown_label(self):
        strategy = FewShotStrategy()
        config = LLMClassifierConfig(
            unknown_label="none",
            examples=[ClassificationExample(text="meow", label="cat")],
        )
        prompt = strategy.build_system_prompt(config, ["cat", "dog"], multi_label=True)
        assert "meow" in prompt
