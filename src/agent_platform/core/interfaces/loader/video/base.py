from abc import abstractmethod
from collections.abc import Sequence
from typing import TypeVar

from agent_platform.core.interfaces.loader.base import BaseMediaLoader
from agent_platform.core.interfaces.loader.video.config import VideoLoaderConfig
from agent_platform.core.schemas.document import VideoDocument

ConfigT = TypeVar("ConfigT", bound=VideoLoaderConfig)


class BaseVideoLoader(BaseMediaLoader[VideoDocument, ConfigT]):
    @abstractmethod
    async def load(
        self, source: str, config: ConfigT | None = None
    ) -> Sequence[VideoDocument]: ...
