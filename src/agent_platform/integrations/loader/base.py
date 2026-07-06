from abc import ABC, abstractmethod
from typing import AsyncIterator, Sequence, TypeVar, Generic

from agent_platform.models.document import Document

T_co = TypeVar("T_co", covariant=True, bound=Document)


class BaseMediaLoader(ABC, Generic[T_co]):

    @abstractmethod
    async def load(self, source: str) -> Sequence[T_co]: ...

    async def load_many(self, sources: list[str], **kwargs) -> AsyncIterator[Sequence[T_co]]:
        for source in sources:
            yield await self.load(source, **kwargs)
