from __future__ import annotations

import logging

__all__ = ["warn_unsupported"]

_logger = logging.getLogger(__name__)


def warn_unsupported(provider: str, field: str) -> None:
    """Logs that a generic `GenerationConfig` field is not sent to `provider`.

    Only the field name is logged, never its value.
    """
    _logger.warning(
        "%s is ignored: the %s adapter does not support it", field, provider
    )
