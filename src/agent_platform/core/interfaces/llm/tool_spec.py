from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

__all__ = ["ToolSpec"]


class ToolSpec(Protocol):
    """What an LLM provider needs to offer a tool to the model.

    Only the declaration: a name, a description and the arguments schema.
    Running the tool belongs to `agents.tools.Tool`, which satisfies this
    protocol structurally, so `core/` and `integrations/` never import it.
    Read-only, so a class attribute (`name = "search"`) satisfies it.
    """

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def input_schema(self) -> type[BaseModel]: ...
