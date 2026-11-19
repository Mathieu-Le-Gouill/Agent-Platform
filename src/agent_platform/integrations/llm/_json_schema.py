from __future__ import annotations

import re
from typing import Any

__all__ = ["json_schema_name"]

_DEFAULT_NAME = "response"
# OpenAI's rule for a `response_format` schema name; Mistral accepts the same.
_INVALID_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
_MAX_NAME_LENGTH = 64


def json_schema_name(schema: dict[str, Any]) -> str:
    """Returns the name to send with a JSON schema response format.

    The name is the schema's `title` reduced to `[a-zA-Z0-9_-]` and at most 64
    characters, or "response" when there is no usable title.
    """
    title = schema.get("title")
    if isinstance(title, str):
        name = _INVALID_NAME_CHARS.sub("", title)[:_MAX_NAME_LENGTH]
        if name:
            return name
    return _DEFAULT_NAME
