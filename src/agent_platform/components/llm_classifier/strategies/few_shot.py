from agent_platform.components.llm_classifier.config import (
    ClassificationMode,
    LLMClassifierConfig,
)
from agent_platform.components.llm_classifier.strategies.prompt_templates import (
    build_classification_template,
)
from agent_platform.components.llm_classifier.strategies.registry import (
    register_strategy,
)


@register_strategy(ClassificationMode.FEW_SHOT)
class FewShotStrategy:
    def build_system_prompt(
        self,
        config: LLMClassifierConfig,
        candidate_labels: list[str],
        *,
        multi_label: bool,
    ) -> str:
        if not config.examples:
            raise ValueError("FEW_SHOT mode requires config.examples")

        examples = "\n\n".join(
            f"Text: {e.text}\nLabel: {e.label}" for e in config.examples
        )

        template = build_classification_template(
            strategy=ClassificationMode.FEW_SHOT,
            multi_label=multi_label,
        )

        values = {
            "labels": ", ".join(candidate_labels),
            "examples": examples,
        }

        if multi_label:
            values["unknown"] = config.unknown_label

        return template.format(**values)
