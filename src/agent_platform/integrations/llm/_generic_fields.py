from __future__ import annotations

import logging
from collections.abc import Mapping

__all__ = ["levels_for_model", "nearest_level", "warn_unsupported"]

_logger = logging.getLogger(__name__)

# Generic reasoning levels, lowest to highest.
_LEVEL_SCALE = ("none", "minimal", "low", "medium", "high", "xhigh")


def warn_unsupported(provider: str, field: str) -> None:
    """Logs that a generic `GenerationConfig` field is not sent to `provider`.

    Only the field name is logged, never its value.
    """
    _logger.warning(
        "%s is ignored: the %s adapter does not support it", field, provider
    )


def levels_for_model(
    model: str, table: Mapping[str, frozenset[str]]
) -> frozenset[str] | None:
    """Returns the table entry of the longest prefix of the lowercased `model`.

    Returns `None` when no prefix matches, so the caller treats the model as
    unknown.
    """
    model_lower = model.lower()
    matches = [prefix for prefix in table if model_lower.startswith(prefix)]
    if not matches:
        return None
    return table[max(matches, key=len)]


def nearest_level(
    provider: str, level: str, supported: frozenset[str], model: str
) -> str:
    """Returns the supported level closest to `level` on the generic scale.

    A tie goes to the higher level. `supported` must not be empty. A changed
    level is logged at debug level: only level names and the model id.
    """
    if level in supported:
        return level
    position = _LEVEL_SCALE.index(level)
    chosen = min(
        supported,
        key=lambda s: (
            abs(_LEVEL_SCALE.index(s) - position),
            -_LEVEL_SCALE.index(s),
        ),
    )
    _logger.debug(
        "%s: reasoning_effort %r mapped to %r for model %r",
        provider,
        level,
        chosen,
        model,
    )
    return chosen
