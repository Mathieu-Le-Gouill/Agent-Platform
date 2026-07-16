from typing import Protocol

from agent_platform.components.llm_classifier.config import LLMClassifierConfig


class PromptStrategy(Protocol):
    def build_system_prompt(
        self,
        config: LLMClassifierConfig,
        candidate_labels: list[str],
        *,
        multi_label: bool,
    ) -> str: ...
