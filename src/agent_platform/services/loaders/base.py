from typing import AsyncIterator, Protocol
from models.protocols.loadable import Loadable


class BaseLoader(Protocol):
    """Loads raw source material and returns a populated Document."""

    async def load(self, source: str, **kwargs) -> Loadable: ...

    async def load_many(self, sources: list[str], **kwargs) -> AsyncIterator[Loadable]: ...