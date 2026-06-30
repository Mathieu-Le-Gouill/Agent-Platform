from abc import abstractmethod
from typing import AsyncIterator, Sequence

from models.document import ImageFile
from integrations.loader.base import BaseMediaLoader


class BaseImageLoader(BaseMediaLoader[ImageFile]):

    @abstractmethod
    async def load(self, source: str) -> Sequence[ImageFile]: ...
    

    @abstractmethod
    def load_many(self, sources: list[str], **kwargs) -> AsyncIterator[Sequence[ImageFile]]: ...