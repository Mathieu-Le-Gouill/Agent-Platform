import pytest
from pydantic import BaseModel, Field

from agent_platform.agents.tools._utils import (
    safe_call,
    audio_chunk,
    text_chunk,
    filter_by_confidence,
    tool_to_openai_schema,
    tool_to_anthropic_schema,
    tool_schema_description,
)
from agent_platform.agents.tools.base import Tool, ToolError
from agent_platform.models.chunk import AudioChunk, TextChunk
from agent_platform.models.enums import DataType


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


# ---------------------------------------------------------------------------
# audio_chunk
# ---------------------------------------------------------------------------


class TestAudioChunk:
    def test_defaults(self):
        chunk = audio_chunk(data=b"\x00\x01")
        assert isinstance(chunk, AudioChunk)
        assert chunk.data == b"\x00\x01"
        assert chunk.sample_rate == 16000
        assert chunk.channels == 1
        assert chunk.dtype == DataType.FLOAT32

    def test_custom_params(self):
        chunk = audio_chunk(data=b"\x00", sample_rate=44100, channels=2)
        assert chunk.sample_rate == 44100
        assert chunk.channels == 2

    def test_unique_ids(self):
        c1 = audio_chunk(b"\x00")
        c2 = audio_chunk(b"\x00")
        assert c1.id != c2.id


# ---------------------------------------------------------------------------
# text_chunk
# ---------------------------------------------------------------------------


class TestTextChunk:
    def test_creates_chunk(self):
        chunk = text_chunk("hello")
        assert isinstance(chunk, TextChunk)
        assert chunk.text == "hello"
        assert chunk.index == 0

    def test_unique_ids(self):
        c1 = text_chunk("a")
        c2 = text_chunk("a")
        assert c1.id != c2.id


# ---------------------------------------------------------------------------
# filter_by_confidence
# ---------------------------------------------------------------------------


class TestFilterByConfidence:
    def make_chunks(self, confidences: list[float]) -> list[TextChunk]:
        return [
            TextChunk(
                text=f"chunk{i}",
                index=i,
                metadata={"confidence": c},
            )
            for i, c in enumerate(confidences)
        ]

    def test_zero_threshold_returns_all(self):
        chunks = self.make_chunks([0.1, 0.5, 0.9])
        result = filter_by_confidence(chunks, min_confidence=0.0)
        assert len(result) == 3

    def test_filters_below_threshold(self):
        chunks = self.make_chunks([0.1, 0.5, 0.9])
        result = filter_by_confidence(chunks, min_confidence=0.5)
        assert len(result) == 2
        assert result[0].text == "chunk1"
        assert result[1].text == "chunk2"

    def test_all_below_returns_empty(self):
        chunks = self.make_chunks([0.1, 0.2])
        result = filter_by_confidence(chunks, min_confidence=0.5)
        assert result == []

    def test_missing_confidence_defaults_to_1(self):
        chunks = [TextChunk(text="no_conf", index=0, metadata={})]
        result = filter_by_confidence(chunks, min_confidence=0.5)
        assert len(result) == 1

    def test_returns_new_list(self):
        chunks = self.make_chunks([0.5])
        result = filter_by_confidence(chunks, min_confidence=0.5)
        assert result is not chunks
        assert result == list(chunks)


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


class TestToolToOpenAISchema:
    def test_structure(self):
        tool = _SchemaTool()
        schema = tool_to_openai_schema(tool)
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"
        assert schema["function"]["description"] == "A test tool for schema generation"

    def test_custom_input_model_properties(self):
        tool = _CustomSchemaTool()
        schema = tool_to_openai_schema(tool)
        props = schema["function"]["parameters"]["properties"]
        assert "name" in props
        assert "count" in props
        assert props["count"]["default"] == 1

    def test_custom_input_model_required(self):
        tool = _CustomSchemaTool()
        schema = tool_to_openai_schema(tool)
        assert "name" in schema["function"]["parameters"]["required"]


class TestToolToAnthropicSchema:
    def test_structure(self):
        tool = _SchemaTool()
        schema = tool_to_anthropic_schema(tool)
        assert schema["name"] == "test_tool"
        assert "input_schema" in schema
        assert "properties" in schema["input_schema"]

    def test_no_function_wrapper(self):
        tool = _SchemaTool()
        schema = tool_to_anthropic_schema(tool)
        assert "type" not in schema
        assert "function" not in schema


class TestToolSchemaDescription:
    def test_openai_provider(self):
        tool = _SchemaTool()
        result = tool_schema_description(tool, provider="openai")
        assert result["type"] == "function"

    def test_anthropic_provider(self):
        tool = _SchemaTool()
        result = tool_schema_description(tool, provider="anthropic")
        assert "input_schema" in result

    def test_unsupported_provider(self):
        tool = _SchemaTool()
        with pytest.raises(ValueError, match="Unsupported provider"):
            tool_schema_description(tool, provider="gemini")


class TestToolBaseSchemaMethods:
    def test_to_openai_schema(self):
        tool = _CustomSchemaTool()
        schema = tool.to_openai_schema()
        assert schema["function"]["name"] == "custom_input"

    def test_to_anthropic_schema(self):
        tool = _CustomSchemaTool()
        schema = tool.to_anthropic_schema()
        assert schema["name"] == "custom_input"
