from openai._types import Omit
from typing import TypeVar

T = TypeVar("T")

def omit_none(value: T | None) -> T | Omit:
    return value if value is not None else Omit()