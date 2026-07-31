from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class MCPToolSpec(BaseModel, frozen=True):
    """One tool advertised by an MCP server, as returned by `list_tools()`.

    `input_schema` is the tool's raw JSON Schema (not a `core/schemas` type),
    since it comes from an arbitrary external server and its shape is only
    known at connect time; `agents/tools/mcp/adapter.py` turns it into a
    concrete Pydantic model per tool.
    """

    name: str
    description: str = ""
    input_schema: dict[str, Any]
