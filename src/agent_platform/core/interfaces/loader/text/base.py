from abc import abstractmethod
from typing import Sequence, TypeVar

from agent_platform.core.credentials import NoCredentials
from agent_platform.core.schemas.document import TextDocument
from agent_platform.core.interfaces.loader.base import BaseMediaLoader
from agent_platform.core.interfaces.loader.config import LoaderConfig

ConfigT = TypeVar("ConfigT", bound=LoaderConfig)


class BaseTextLoader(BaseMediaLoader[NoCredentials, TextDocument, ConfigT]):
    @abstractmethod
    async def load(
        self, source: str, config: ConfigT | None = None
    ) -> Sequence[TextDocument]: ...
