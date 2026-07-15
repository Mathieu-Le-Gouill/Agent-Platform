from abc import ABC, abstractmethod
from typing import AsyncIterator, Sequence, TypeVar, Generic
import asyncio

from agent_platform.core.credentials import BaseCredentials
from agent_platform.core.interfaces.loader.config import LoaderConfig
from agent_platform.core.schemas.document import Document

T_co = TypeVar("T_co", covariant=True, bound=Document)
ConfigT = TypeVar("ConfigT", bound=LoaderConfig)
CredentialsT = TypeVar("CredentialsT", bound=BaseCredentials)


class BaseMediaLoader(ABC, Generic[CredentialsT, T_co, ConfigT]):
    def __init__(self, credentials: CredentialsT) -> None:
        self._credentials = credentials

    @abstractmethod
    async def load(
        self, source: str, config: ConfigT | None = None
    ) -> Sequence[T_co]: ...

    async def load_many(
        self,
        sources: list[str],
        config: ConfigT | None = None,
    ) -> AsyncIterator[Sequence[T_co]]:
        results = await asyncio.gather(
            *[self.load(s, config) for s in sources],
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                continue
            yield result
