import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Generic, TypeVar

from agent_platform.core.interfaces.loader.config import LoaderConfig
from agent_platform.core.schemas.document import Document

DocumentT = TypeVar("DocumentT", covariant=True, bound=Document)
ConfigT = TypeVar("ConfigT", bound=LoaderConfig)


class BaseMediaLoader(ABC, Generic[DocumentT, ConfigT]):
    @abstractmethod
    async def load(
        self, source: str, config: ConfigT | None = None
    ) -> Sequence[DocumentT]: ...

    async def load_many(
        self,
        sources: list[str],
        config: ConfigT | None = None,
    ) -> AsyncIterator[Sequence[DocumentT]]:
        results = await asyncio.gather(
            *[self.load(s, config) for s in sources],
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                continue  # a failed source is skipped silently, not raised
            yield result
