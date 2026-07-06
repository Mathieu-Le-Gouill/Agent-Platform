from abc import abstractmethod
from typing import AsyncIterator, Sequence

from agent_platform.models.document import TextDocument
from agent_platform.integrations.loader.base import BaseMediaLoader


class BaseTextLoader(BaseMediaLoader[TextDocument]):
    @abstractmethod
    async def load(self, source: str) -> Sequence[TextDocument]: ...

    @abstractmethod
    def load_many(
        self, sources: list[str], **kwargs
    ) -> AsyncIterator[Sequence[TextDocument]]: ...
