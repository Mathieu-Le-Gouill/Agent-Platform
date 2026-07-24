from abc import abstractmethod
from collections.abc import Sequence
from typing import TypeVar

from agent_platform.core.interfaces.loader.base import BaseMediaLoader
from agent_platform.core.interfaces.loader.config import LoaderConfig
from agent_platform.core.schemas.document import TextDocument

ConfigT = TypeVar("ConfigT", bound=LoaderConfig)


class BaseTextLoader(BaseMediaLoader[TextDocument, ConfigT]):
    @abstractmethod
    async def load(
        self, source: str, config: ConfigT | None = None
    ) -> Sequence[TextDocument]: ...
