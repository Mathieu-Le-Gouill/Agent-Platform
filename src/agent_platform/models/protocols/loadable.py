from typing import Protocol, runtime_checkable
from typing import Optional


@runtime_checkable
class Loadable(Protocol):
    @property
    def source(self) -> str: ...

    @property
    def mime_type(self) -> Optional[str]: ...