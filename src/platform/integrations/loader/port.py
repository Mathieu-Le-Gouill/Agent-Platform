from typing import AsyncIterator, Protocol, Sequence
from models.protocols.loadable import Loadable


class MediaLoader(Protocol):

    async def load(self, source: str, **kwargs) -> Sequence[Loadable]: ...
    

    async def load_many(self, sources: list[str], **kwargs) -> AsyncIterator[Sequence[Loadable]]: ...