from abc import abstractmethod
from typing import Sequence, TypeVar

from agent_platform.core.credentials import NoCredentials
from agent_platform.core.schemas.document import VideoDocument
from agent_platform.core.interfaces.loader.base import BaseMediaLoader
from agent_platform.core.interfaces.loader.video.config import VideoLoaderConfig

ConfigT = TypeVar("ConfigT", bound=VideoLoaderConfig)


class BaseVideoLoader(BaseMediaLoader[NoCredentials, VideoDocument, ConfigT]):
    @abstractmethod
    async def load(
        self, source: str, config: ConfigT | None = None
    ) -> Sequence[VideoDocument]: ...
