from typing import Any

import pytest
from pydantic import ValidationError

from agent_platform.agents.tools.base import ToolError
from agent_platform.agents.tools.mcp.adapter import (
    MCPToolAdapter,
    _input_schema_from_json_schema,
)
from agent_platform.core.interfaces.mcp.base import BaseMCPClient
from agent_platform.core.schemas.mcp import MCPToolSpec


class _FakeMCPClient(BaseMCPClient):
    def __init__(self, result: Any = "ok", *, raises: Exception | None = None) -> None:
        self.result = result
        self.raises = raises
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def connect(self) -> None:
        pass

    async def aclose(self) -> None:
        pass

    async def list_tools(self, config=None):
        return []

    async def call_tool(self, name, arguments, config=None):
        self.calls.append((name, arguments))
        if self.raises is not None:
            raise self.raises
        return self.result


_ECHO_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "count": {"type": "integer"},
        "loud": {"type": "boolean"},
    },
    "required": ["text"],
}


class TestInputSchemaFromJsonSchema:
    def test_required_field_is_required(self):
        model = _input_schema_from_json_schema("echo", _ECHO_SCHEMA)
        with pytest.raises(ValidationError):
            model()

    def test_optional_fields_default_to_none(self):
        model = _input_schema_from_json_schema("echo", _ECHO_SCHEMA)
        instance = model(text="hi")
        assert instance.count is None
        assert instance.loud is None

    def test_valid_instance_round_trips(self):
        model = _input_schema_from_json_schema("echo", _ECHO_SCHEMA)
        instance = model(text="hi", count=3, loud=True)
        assert instance.text == "hi"
        assert instance.count == 3
        assert instance.loud is True

    def test_unknown_type_falls_back_to_any(self):
        schema = {
            "type": "object",
            "properties": {"payload": {"type": "unknown_type"}},
            "required": ["payload"],
        }
        model = _input_schema_from_json_schema("weird", schema)
        instance = model(payload={"anything": [1, 2, 3]})
        assert instance.payload == {"anything": [1, 2, 3]}

    def test_no_properties_produces_empty_model(self):
        model = _input_schema_from_json_schema("noop", {"type": "object"})
        instance = model()
        assert instance.model_dump() == {}


class TestMCPToolAdapter:
    def test_name_and_description_from_spec(self):
        client = _FakeMCPClient()
        spec = MCPToolSpec(
            name="echo", description="Echoes text", input_schema=_ECHO_SCHEMA
        )
        tool = MCPToolAdapter(client, spec)
        assert tool.name == "echo"
        assert tool.description == "Echoes text"

    def test_description_falls_back_when_spec_has_none(self):
        client = _FakeMCPClient()
        spec = MCPToolSpec(name="echo", input_schema=_ECHO_SCHEMA)
        tool = MCPToolAdapter(client, spec)
        assert tool.description == "MCP tool 'echo'"

    @pytest.mark.asyncio
    async def test_run_delegates_to_client(self):
        client = _FakeMCPClient(result="done")
        spec = MCPToolSpec(name="echo", input_schema=_ECHO_SCHEMA)
        tool = MCPToolAdapter(client, spec)

        result = await tool.run(text="hi")

        assert result == "done"
        assert client.calls == [("echo", {"text": "hi"})]

    @pytest.mark.asyncio
    async def test_run_wraps_client_errors_as_tool_error(self):
        client = _FakeMCPClient(raises=RuntimeError("boom"))
        spec = MCPToolSpec(name="echo", input_schema=_ECHO_SCHEMA)
        tool = MCPToolAdapter(client, spec)

        with pytest.raises(ToolError):
            await tool.run(text="hi")
