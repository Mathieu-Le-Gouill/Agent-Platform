from abc import abstractmethod
from typing import Sequence, TypeVar

from agent_platform.core.schemas.document import ImageDocument
from agent_platform.core.interfaces.loader.base import BaseMediaLoader
from agent_platform.core.interfaces.loader.image.config import ImageLoaderConfig

ConfigT = TypeVar("ConfigT", bound=ImageLoaderConfig)


class BaseImageLoader(BaseMediaLoader[ImageDocument, ConfigT]):
    @abstractmethod
    async def load(
        self, source: str, config: ConfigT | None = None
    ) -> Sequence[ImageDocument]: ...
