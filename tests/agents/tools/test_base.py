import pytest
from pydantic import BaseModel

from agent_platform.agents.tools.base import Tool, ToolError


class _DefaultAstreamTool(Tool):
    name = "default_astream"
    description = "Uses the default astream implementation"
    input_schema = BaseModel

    async def run(self, **kwargs):
        return None


def test_tool_cannot_instantiate_abc():
    with pytest.raises(TypeError):
        Tool()  # type: ignore[abstract]


def test_subclass_must_define_name():
    with pytest.raises(ToolError, match="must define a non-empty 'name'"):

        class NoName(Tool):
            description = "desc"
            input_schema = BaseModel

            async def run(self, **kwargs):
                return None


def test_subclass_must_define_description():
    with pytest.raises(ToolError, match="must define a non-empty 'description'"):

        class NoDesc(Tool):
            name = "test"
            input_schema = BaseModel

            async def run(self, **kwargs):
                return None


def test_subclass_empty_name_is_rejected():
    with pytest.raises(ToolError, match="must define a non-empty 'name'"):

        class EmptyName(Tool):
            name = ""
            description = "desc"
            input_schema = BaseModel

            async def run(self, **kwargs):
                return None


def test_concrete_tool_instantiates():
    class Concrete(Tool):
        name = "test_tool"
        description = "A test tool"
        input_schema = BaseModel

        async def run(self, **kwargs):
            return {"result": "ok"}

    instance = Concrete()
    assert instance.name == "test_tool"
    assert instance.description == "A test tool"
    assert instance.input_schema is BaseModel


def test_run_executes_successfully():
    class Echo(Tool):
        name = "echo"
        description = "Echoes input"
        input_schema = BaseModel

        async def run(self, **kwargs):
            return kwargs

    instance = Echo()
    import asyncio

    result = asyncio.run(instance.run(key="value"))
    assert result == {"key": "value"}


@pytest.mark.asyncio
async def test_run_async():
    class AsyncTool(Tool):
        name = "async_tool"
        description = "Async tool"
        input_schema = BaseModel

        async def run(self, **kwargs):
            return {"async": True}

    instance = AsyncTool()
    result = await instance.run()
    assert result == {"async": True}


def test_output_schema_defaults_to_none():
    class NoOutput(Tool):
        name = "no_output"
        description = "desc"
        input_schema = BaseModel

        async def run(self, **kwargs):
            return None

    instance = NoOutput()
    assert instance.output_schema is None


def test_output_schema_can_be_set():
    class OutputModel(BaseModel):
        value: str

    class WithOutput(Tool):
        name = "with_output"
        description = "desc"
        input_schema = BaseModel
        output_schema = OutputModel

        async def run(self, **kwargs):
            return OutputModel(value="ok")

    assert WithOutput.output_schema is OutputModel


@pytest.mark.asyncio
async def test_default_astream_raises_not_implemented():
    instance = _DefaultAstreamTool()
    with pytest.raises(NotImplementedError, match="does not support streaming"):
        async for _ in instance.astream():
            pass
