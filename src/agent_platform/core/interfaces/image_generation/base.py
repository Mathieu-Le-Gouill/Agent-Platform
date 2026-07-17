from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from agent_platform.core.interfaces.image_generation.config import ImageGenConfig
from agent_platform.core.schemas.document import ImageDocument
from agent_platform.core.schemas.enums import ImageFormat

ConfigT = TypeVar("ConfigT", bound=ImageGenConfig, covariant=True)


class BaseImageGenerator(ABC, Generic[ConfigT]):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        config: ConfigT | None = None,
        size: str | None = None,
        format: ImageFormat = ImageFormat.PNG,
    ) -> ImageDocument: ...

    @abstractmethod
    async def generate_many(
        self,
        prompt: str,
        n: int = 1,
        config: ConfigT | None = None,
        size: str | None = None,
        format: ImageFormat = ImageFormat.PNG,
    ) -> list[ImageDocument]: ...
