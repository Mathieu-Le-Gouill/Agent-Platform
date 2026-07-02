from abc import ABC, abstractmethod
from typing import AsyncIterator, Sequence, TypeVar, Generic

from agent_platform.models.protocols.loadable import Loadable


T_co = TypeVar("T_co", bound=Loadable, covariant=True)

class BaseMediaLoader(ABC, Generic[T_co]):

    @abstractmethod
    async def load(self, source: str) -> Sequence[T_co]: ...
    

    @abstractmethod
    def load_many(self, sources: list[str], **kwargs) -> AsyncIterator[Sequence[T_co]]: ...