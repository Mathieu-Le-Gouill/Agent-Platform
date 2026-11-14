from __future__ import annotations

import hashlib
import json
import os
from typing import Any, NamedTuple

__all__ = [
    "PayloadFacts",
    "capture_content",
    "payload_facts",
]

_CAPTURE_ENV_KEY = "AGENT_PLATFORM_CAPTURE_CONTENT"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_DIGEST_LENGTH = 16


class PayloadFacts(NamedTuple):
    """What telemetry may say about a payload.

    `size` and `digest` are always safe to export: they answer "how big was
    it" and "was it the same value as last time" without revealing content.
    `content` is the serialized payload itself, and is `None` unless content
    capture is explicitly enabled.
    """

    size: int
    digest: str
    content: str | None


def capture_content() -> bool:
    """Whether telemetry may carry prompt/tool payloads verbatim.

    Off by default: payloads routinely contain user data, and a tracing
    backend is not a place it should land without someone deciding so. Read
    per call rather than cached, so toggling it never needs a restart.
    """
    value = os.environ.get(_CAPTURE_ENV_KEY)
    return value is not None and value.strip().lower() in _TRUE_VALUES


def payload_facts(value: Any) -> PayloadFacts:
    """Summarize `value`, honoring `capture_content()`.

    Strings are measured as-is; anything else is JSON-serialized first
    (falling back to `str` for values JSON can't represent, since a payload
    summary must never be the thing that fails an observed call).
    """
    serialized = value if isinstance(value, str) else _to_json(value)
    encoded = serialized.encode("utf-8")
    return PayloadFacts(
        size=len(encoded),
        digest=hashlib.sha256(encoded).hexdigest()[:_DIGEST_LENGTH],
        content=serialized if capture_content() else None,
    )


def _to_json(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True)
