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


@register_strategy(ClassificationMode.TEXT_CLASSIFICATION)
class TextClassificationStrategy:
    def build_system_prompt(
        self,
        config: LLMClassifierConfig,
        candidate_labels: list[str],
        multi_label: bool,
    ) -> str:
        template = build_classification_template(
            strategy=ClassificationMode.TEXT_CLASSIFICATION,
            multi_label=multi_label,
        )

        values = {"labels": ", ".join(candidate_labels)}

        if not multi_label:
            values["unknown"] = config.unknown_label

        return template.format(**values)
