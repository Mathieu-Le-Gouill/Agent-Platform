from __future__ import annotations

import functools
from typing import Any

from pydantic import BaseModel


@functools.lru_cache(maxsize=128)
def model_schema(model: type[BaseModel]) -> dict[str, Any]:
    if model is BaseModel:
        return {"type": "object", "properties": {}}
    return model.model_json_schema()
