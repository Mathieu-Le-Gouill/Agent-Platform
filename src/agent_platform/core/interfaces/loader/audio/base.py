from abc import abstractmethod
from collections.abc import Sequence
from typing import TypeVar

from agent_platform.core.interfaces.loader.audio.config import AudioLoaderConfig
from agent_platform.core.interfaces.loader.base import BaseMediaLoader
from agent_platform.core.schemas.document import AudioDocument

ConfigT = TypeVar("ConfigT", bound=AudioLoaderConfig)


class BaseAudioLoader(BaseMediaLoader[AudioDocument, ConfigT]):
    @abstractmethod
    async def load(
        self, source: str, config: ConfigT | None = None
    ) -> Sequence[AudioDocument]: ...
