from typing import Protocol, runtime_checkable


@runtime_checkable
class Loadable(Protocol):
    @property
    def source(self) -> str: ...

    @property
    def mime_type(self) -> str | None: ...