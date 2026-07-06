from __future__ import annotations

from abc import ABC, abstractmethod

from agent_platform.models.document import ImageDocument
from agent_platform.models.enums import ImageFormat


class BaseImageGenerator(ABC):

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        size: str | None = None,
        format: ImageFormat = ImageFormat.PNG,
    ) -> ImageDocument: ...

    @abstractmethod
    async def generate_many(
        self,
        prompt: str,
        n: int = 1,
        size: str | None = None,
        format: ImageFormat = ImageFormat.PNG,
    ) -> list[ImageDocument]: ...
