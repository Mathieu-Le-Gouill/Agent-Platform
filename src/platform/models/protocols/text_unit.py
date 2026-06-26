from typing import Protocol, runtime_checkable


@runtime_checkable
class TextUnit(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def text(self) -> str: ...