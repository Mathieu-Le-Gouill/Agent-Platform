from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

from pydantic import BaseModel

from agent_platform.agents.tools.errors import ToolError as ToolError


class ToolStreamChunk(BaseModel, frozen=True):
    tool_call_id: str
    delta: str
    is_final: bool = False
    is_error: bool = False
    is_validation_error: bool = False


class Tool(Protocol):
    name: str = ""
    description: str = ""
    input_schema: type[BaseModel] = BaseModel
    output_schema: type[BaseModel] | None = None
    supports_streaming: bool = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not cls.name:
            raise ToolError(f"{cls.__name__} must define a non-empty 'name'")
        if not cls.description:
            raise ToolError(f"{cls.__name__} must define a non-empty 'description'")

    async def run(self, **kwargs: Any) -> Any: ...

    async def astream(self, **kwargs: Any) -> AsyncIterator[str]:
        raise NotImplementedError(f"{type(self).__name__} does not support streaming")
        yield  # pragma: no cover - marks this as an async generator
