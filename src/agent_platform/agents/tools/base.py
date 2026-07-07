from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from agent_platform.core.errors import PlatformError


class ToolError(PlatformError):
    pass


class Tool(ABC):
    name: str = ""
    description: str = ""
    input_schema: type[BaseModel] = BaseModel
    output_schema: type[BaseModel] | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not cls.name:
            raise ToolError(f"{cls.__name__} must define a non-empty 'name'")
        if not cls.description:
            raise ToolError(f"{cls.__name__} must define a non-empty 'description'")

    @abstractmethod
    async def run(self, **kwargs: Any) -> Any: ...
