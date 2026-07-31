import pytest
from pydantic import ValidationError

from agent_platform.core.schemas.mcp import MCPToolSpec


class TestMCPToolSpec:
    def test_defaults(self):
        spec = MCPToolSpec(name="echo", input_schema={"type": "object"})
        assert spec.name == "echo"
        assert spec.description == ""
        assert spec.input_schema == {"type": "object"}

    def test_is_frozen(self):
        spec = MCPToolSpec(name="echo", input_schema={})
        with pytest.raises(ValidationError):
            spec.name = "other"
