import pytest
from pydantic import BaseModel, Field

from agent_platform.agents.tools.base import Tool, ToolError
from agent_platform.agents.tools.safe_execution import safe_call, safe_stream
from agent_platform.core.schemas import model_schema

# ---------------------------------------------------------------------------
# safe_call
# ---------------------------------------------------------------------------


class TestSafeCall:
    @pytest.mark.asyncio
    async def test_success_returns_value(self):
        async def ok():
            return 42

        result = await safe_call(ok())
        assert result == 42

    @pytest.mark.asyncio
    async def test_tool_error_passes_through(self):
        async def raises_tool_error():
            raise ToolError("already wrapped")

        with pytest.raises(ToolError, match="already wrapped"):
            await safe_call(raises_tool_error())

    @pytest.mark.asyncio
    async def test_runtime_error_wrapped(self):
        async def raises_raw():
            raise ValueError("original")

        with pytest.raises(ToolError, match="custom message: original"):
            await safe_call(raises_raw(), error_message="custom message")

    @pytest.mark.asyncio
    async def test_custom_error_message(self):
        async def raises():
            raise RuntimeError("fail")

        with pytest.raises(ToolError, match="Provider execution failed"):
            await safe_call(raises())


class TestSafeStream:
    @pytest.mark.asyncio
    async def test_yields_all_items(self):
        async def gen():
            for i in range(3):
                yield i

        result = [item async for item in safe_stream(gen())]
        assert result == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_tool_error_passes_through(self):
        async def gen():
            raise ToolError("already wrapped")
            yield  # pragma: no cover

        with pytest.raises(ToolError, match="already wrapped"):
            async for _ in safe_stream(gen()):
                pass

    @pytest.mark.asyncio
    async def test_raw_exception_wrapped(self):
        async def gen():
            yield "partial"
            raise ValueError("stream broke")

        collected = []
        with pytest.raises(ToolError, match="custom stream failure: stream broke"):
            async for item in safe_stream(gen(), error_message="custom stream failure"):
                collected.append(item)
        assert collected == ["partial"]

    @pytest.mark.asyncio
    async def test_default_error_message(self):
        async def gen():
            raise RuntimeError("fail")
            yield  # pragma: no cover

        with pytest.raises(ToolError, match="Provider execution failed"):
            async for _ in safe_stream(gen()):
                pass


# ---------------------------------------------------------------------------
# Schema generation
# ---------------------------------------------------------------------------


class _SchemaTool(Tool):
    name = "test_tool"
    description = "A test tool for schema generation"
    input_schema = BaseModel
    output_schema = None

    async def run(self, **kwargs):
        return None


class _InputModel(BaseModel):
    name: str = Field(..., description="The name")
    count: int = Field(default=1, ge=0, description="Number of items")


class _CustomSchemaTool(Tool):
    name = "custom_input"
    description = "Tool with custom input schema"
    input_schema = _InputModel
    output_schema = None

    async def run(self, **kwargs):
        return None


class TestModelSchema:
    def test_default_base_model(self):
        schema = model_schema(BaseModel)
        assert schema == {"type": "object", "properties": {}}

    def test_custom_model_properties(self):
        schema = model_schema(_InputModel)
        assert "name" in schema["properties"]
        assert "count" in schema["properties"]
        assert schema["properties"]["count"]["default"] == 1

    def test_custom_model_required(self):
        schema = model_schema(_InputModel)
        assert "name" in schema["required"]

    def test_is_cached(self):
        s1 = model_schema(_InputModel)
        s2 = model_schema(_InputModel)
        assert s1 is s2
