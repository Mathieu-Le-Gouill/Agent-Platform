from __future__ import annotations

from typing import Any

from pydantic import BaseModel, create_model

from agent_platform.agents.tools.base import Tool
from agent_platform.agents.tools.safe_execution import safe_call
from agent_platform.core.interfaces.mcp.base import BaseMCPClient
from agent_platform.core.schemas.mcp import MCPToolSpec

_JSON_SCHEMA_TYPES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _input_schema_from_json_schema(
    tool_name: str, schema: dict[str, Any]
) -> type[BaseModel]:
    """Build a Pydantic model from an MCP tool's raw JSON Schema.

    Covers the common case (a flat `object` schema with primitive/array/object
    typed properties), since that's what MCP tool schemas are in practice; an
    unrecognized `type` falls back to `Any` rather than raising, so an unusual
    schema still produces a usable (permissive) model instead of blocking
    registration.
    """
    properties: dict[str, Any] = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {}
    for field_name, field_schema in properties.items():
        python_type = _JSON_SCHEMA_TYPES.get(field_schema.get("type"), Any)
        if field_name in required:
            fields[field_name] = (python_type, ...)
        else:
            fields[field_name] = (python_type | None, None)
    model_name = "".join(part.title() for part in tool_name.split("_")) + "Input"
    return create_model(model_name, **fields)


class MCPToolAdapter(Tool):
    """Exposes one MCP server tool as a platform `Tool`.

    `name`/`description`/`input_schema` are overridden per instance from the
    discovered `MCPToolSpec` (the class-level defaults below only exist to
    satisfy `Tool.__init_subclass__`'s non-empty check, which runs once at
    class definition time, before any spec exists); execution is delegated to
    `client.call_tool()`, so this adapter carries no MCP-transport knowledge.
    """

    name = "mcp_tool"
    description = "Adapter for a single MCP server tool"

    def __init__(self, client: BaseMCPClient, spec: MCPToolSpec) -> None:
        self.name = spec.name
        self.description = spec.description or f"MCP tool '{spec.name}'"
        self.input_schema = _input_schema_from_json_schema(spec.name, spec.input_schema)
        self._client = client
        self._spec = spec

    async def run(self, **kwargs: Any) -> Any:
        return await safe_call(
            self._client.call_tool(self._spec.name, kwargs),
            f"MCP tool '{self._spec.name}' execution failed",
        )
