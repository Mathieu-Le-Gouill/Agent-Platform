from abc import abstractmethod
from typing import Sequence, TypeVar

from agent_platform.core.schemas.document import AudioDocument
from agent_platform.core.interfaces.loader.base import BaseMediaLoader
from agent_platform.core.interfaces.loader.audio.config import AudioLoaderConfig

ConfigT = TypeVar("ConfigT", bound=AudioLoaderConfig)


class BaseAudioLoader(BaseMediaLoader[AudioDocument, ConfigT]):
    @abstractmethod
    async def load(
        self, source: str, config: ConfigT | None = None
    ) -> Sequence[AudioDocument]: ...
