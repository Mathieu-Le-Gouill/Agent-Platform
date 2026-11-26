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

    `size` is always exported: it reveals only the payload's length, which
    answers "how big was it". `digest` (a short unsalted sha256) and `content` (the
    serialized payload itself) are `None` unless content capture is
    explicitly enabled: a digest of a small value space, such as an email or
    a yes/no, can be brute-forced and links equal values across users, so it
    is as sensitive as the content.
    """

    size: int
    digest: str | None
    content: str | None


def capture_content() -> bool:
    """Whether telemetry may carry prompt/tool payloads verbatim and their digest.

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
    if not capture_content():
        return PayloadFacts(size=len(encoded), digest=None, content=None)
    return PayloadFacts(
        size=len(encoded),
        digest=hashlib.sha256(encoded).hexdigest()[:_DIGEST_LENGTH],
        content=serialized,
    )


def _to_json(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True)
