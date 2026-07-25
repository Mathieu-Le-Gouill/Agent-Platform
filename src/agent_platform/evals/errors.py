from __future__ import annotations

from agent_platform.core.errors import PlatformError

__all__ = [
    "EvalError",
    "EvalDatasetError",
    "EvalRunError",
]


class EvalError(PlatformError):
    pass


class EvalDatasetError(EvalError):
    pass


class EvalRunError(EvalError):
    pass
