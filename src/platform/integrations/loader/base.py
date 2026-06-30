from abc import ABC, abstractmethod
from typing import AsyncIterator, Sequence, TypeVar, Generic

from models.protocols.loadable import Loadable


T = TypeVar("T", bound=Loadable)

class BaseMediaLoader(ABC, Generic[T]):

    @abstractmethod
    async def load(self, source: str) -> Sequence[T]: ...
    

    @abstractmethod
    def load_many(self, sources: list[str], **kwargs) -> AsyncIterator[Sequence[T]]: ...