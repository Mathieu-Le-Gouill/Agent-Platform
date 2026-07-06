from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from agent_platform.core.errors import PlatformError
from agent_platform.agents.tools._utils import (
    tool_to_openai_schema,
    tool_to_anthropic_schema,
)


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

    def to_openai_schema(self) -> dict[str, Any]:
        return tool_to_openai_schema(self)

    def to_anthropic_schema(self) -> dict[str, Any]:
        return tool_to_anthropic_schema(self)
