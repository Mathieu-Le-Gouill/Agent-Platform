from abc import abstractmethod
from typing import AsyncIterator, Sequence

from models.document import TextFile
from integrations.loader.base import BaseMediaLoader


class BaseTextLoader(BaseMediaLoader[TextFile]):

    @abstractmethod
    async def load(self, source: str) -> Sequence[TextFile]: ...
    

    @abstractmethod
    def load_many(self, sources: list[str], **kwargs) -> AsyncIterator[Sequence[TextFile]]: ...