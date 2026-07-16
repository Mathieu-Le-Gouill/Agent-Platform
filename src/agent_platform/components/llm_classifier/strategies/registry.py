from agent_platform.components.llm_classifier.strategies.base import PromptStrategy
from agent_platform.components.llm_classifier.config import ClassificationMode

_STRATEGIES: dict[ClassificationMode, PromptStrategy] = {}


def register_strategy(mode: ClassificationMode):
    def decorator(cls):
        _STRATEGIES[mode] = cls()
        return cls

    return decorator


def get_strategy(mode: ClassificationMode) -> PromptStrategy:
    try:
        return _STRATEGIES[mode]
    except KeyError:
        raise ValueError(f"No strategy registered for mode: {mode}") from None
