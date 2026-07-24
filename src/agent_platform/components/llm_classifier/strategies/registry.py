from collections.abc import Callable
from typing import TypeVar

from agent_platform.components.llm_classifier.config import ClassificationMode
from agent_platform.components.llm_classifier.strategies.base import PromptStrategy

_StrategyT = TypeVar("_StrategyT", bound=type[PromptStrategy])

_STRATEGIES: dict[ClassificationMode, PromptStrategy] = {}


def register_strategy(
    mode: ClassificationMode,
) -> Callable[[_StrategyT], _StrategyT]:
    def decorator(cls: _StrategyT) -> _StrategyT:
        _STRATEGIES[mode] = cls()
        return cls

    return decorator


def get_strategy(mode: ClassificationMode) -> PromptStrategy:
    try:
        return _STRATEGIES[mode]
    except KeyError:
        raise ValueError(f"No strategy registered for mode: {mode}") from None
