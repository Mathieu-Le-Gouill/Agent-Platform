from typing import Protocol, runtime_checkable
from uuid import UUID


@runtime_checkable
class TextUnit(Protocol):
    @property
    def id(self) -> UUID: ...

    @property
    def text(self) -> str: ...