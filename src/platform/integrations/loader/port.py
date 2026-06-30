from typing import AsyncIterator, Protocol, Sequence, TypeVar
from models.protocols.loadable import Loadable


T_co = TypeVar("T_co", bound=Loadable, covariant=True)

class MediaLoader(Protocol[T_co]):

    async def load(self, source: str, **kwargs) -> Sequence[T_co]: ...
    

    async def load_many(self, sources: list[str], **kwargs) -> AsyncIterator[Sequence[T_co]]: ...