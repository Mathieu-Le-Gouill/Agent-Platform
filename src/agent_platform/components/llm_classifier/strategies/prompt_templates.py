from agent_platform.components.llm_classifier.config import ClassificationMode

_BASE_INSTRUCTIONS = (
    "Classify the following text into exactly ONE category from the list below:\n"
    "{labels}\n\n"
    "Rules:\n"
    "- Select exactly one category from the provided list.\n"
    "- Base your decision only on the content of the text.\n"
    "- Do not use outside knowledge.\n"
    "- If the text does not clearly belong to any listed category, "
    'respond with "{unknown}".\n'
)

_BASE_INSTRUCTIONS_MULTI_LABELS = (
    "Classify the following text into zero, one, or multiple categories "
    "from the list below:\n"
    "{labels}\n\n"
    "Rules:\n"
    "- Select every category that is clearly supported by the text.\n"
    "- Categories are independent and multiple categories may apply.\n"
    "- Do not select categories based on weak assumptions.\n"
    "- Base your decision only on the content of the text.\n"
    "- Do not invent categories that are not in the list.\n"
    "- If no category applies, return an empty list.\n"
)

_TEXT_CLASSIFICATION_PREAMBLE = (
    "You are a text-classification system. "
    "Judge only the topic or subject matter of the text, not its tone.\n\n"
)

_SENTIMENT_PREAMBLE = (
    "You are a sentiment-analysis system. "
    "Judge only the emotional tone conveyed by the text \u2014 ignore topic, "
    "factual accuracy, and grammar.\n\n"
)

_FEW_SHOT_EXAMPLES_TEMPLATE = "Examples:\n{examples}\n\n"

_SINGLE_LABEL_TOOL_CALL_INSTRUCTION = (
    "\nCall the `record_classification` function with exactly one category "
    "as the classification result. "
    "The result can be one of the provided categories or the unknown label "
    "if no category applies. "
    "Do not output explanations or additional text."
)

_MULTI_LABEL_TOOL_CALL_INSTRUCTION = (
    "\nCall the `record_classification` function with a list of selected "
    "categories as the classification result. "
    "Return an empty list if no category applies. "
    "Do not output explanations or additional text."
)


def build_classification_template(
    *,
    strategy: ClassificationMode,
    multi_label: bool = False,
) -> str:
    parts: list[str] = []

    match strategy:
        case ClassificationMode.SENTIMENT_ANALYSIS:
            parts.append(_SENTIMENT_PREAMBLE)

        case ClassificationMode.TEXT_CLASSIFICATION:
            parts.append(_TEXT_CLASSIFICATION_PREAMBLE)

        case ClassificationMode.ZERO_SHOT | ClassificationMode.FEW_SHOT:
            pass

        case _:
            raise ValueError(f"Unsupported classification strategy: {strategy}")

    parts.append(_BASE_INSTRUCTIONS_MULTI_LABELS if multi_label else _BASE_INSTRUCTIONS)

    if strategy == ClassificationMode.FEW_SHOT:
        parts.append(_FEW_SHOT_EXAMPLES_TEMPLATE)

    if multi_label:
        parts.append(_MULTI_LABEL_TOOL_CALL_INSTRUCTION)
    else:
        parts.append(_SINGLE_LABEL_TOOL_CALL_INSTRUCTION)

    return "".join(parts)
