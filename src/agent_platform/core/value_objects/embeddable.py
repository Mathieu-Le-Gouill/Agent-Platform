from typing import Protocol, runtime_checkable

@runtime_checkable
class Embeddable(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def content(self) -> str: ...