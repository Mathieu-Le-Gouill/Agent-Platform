from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel

from agent_platform.agents.tools.errors import ToolError as ToolError


class Tool(Protocol):
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

    async def run(self, **kwargs: Any) -> Any: ...
