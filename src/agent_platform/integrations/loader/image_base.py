from abc import abstractmethod
from typing import AsyncIterator, Sequence

from agent_platform.models.document import ImageDocument
from agent_platform.integrations.loader.base import BaseMediaLoader


class BaseImageLoader(BaseMediaLoader[ImageDocument]):

    @abstractmethod
    async def load(self, source: str) -> Sequence[ImageDocument]: ...
    

    @abstractmethod
    def load_many(self, sources: list[str], **kwargs) -> AsyncIterator[Sequence[ImageDocument]]: ...