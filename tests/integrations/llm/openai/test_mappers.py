import copy

import pytest
from pydantic import BaseModel

from agent_platform.core.interfaces.llm.response import ResponseFormat
from agent_platform.core.schemas import model_schema
from agent_platform.core.schemas.document import AudioDocument, ImageDocument
from agent_platform.core.schemas.enums import AudioFormat, FinishReason, ImageFormat
from agent_platform.core.schemas.message import (
    AssistantMessage,
    AudioBlock,
    ImageBlock,
    Prompt,
    SystemMessage,
    TextBlock,
    ToolCall,
    ToolMessage,
    ToolResult,
    UserMessage,
)
from agent_platform.integrations.llm.openai.config import OpenAIGenerationConfig
from agent_platform.integrations.llm.openai.mappers import (
    block_to_native as _block_to_native,
)
from agent_platform.integrations.llm.openai.mappers import (
    from_native_response as _from_native_response,
)
from agent_platform.integrations.llm.openai.mappers import (
    strict_json_schema,
)
from agent_platform.integrations.llm.openai.mappers import (
    to_native_messages as _to_native_messages,
)
from agent_platform.integrations.llm.openai.mappers import (
    to_native_params as _to_native_params,
)
from tests.helpers import make_chat_completion, make_chat_tool_call


class TestBlockToNative:
    def test_text_block(self):
        assert _block_to_native(TextBlock(text="hello")) == {
            "type": "text",
            "text": "hello",
        }

    def test_image_block_url(self):
        result = _block_to_native(ImageBlock(image="https://example.com/a.png"))
        assert result == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/a.png"},
        }

    def test_image_block_document_base64(self):
        doc = ImageDocument(content=b"\x89PNG", format=ImageFormat.PNG)
        result = _block_to_native(ImageBlock(image=doc))
        assert result["type"] == "image_url"
        assert result["image_url"]["url"].startswith("data:image/png;base64,")

    def test_audio_block(self):
        doc = AudioDocument(content=b"RIFF....", format=AudioFormat.WAV)
        result = _block_to_native(AudioBlock(audio=doc))
        assert result["type"] == "input_audio"
        assert result["input_audio"]["format"] == "wav"
        assert result["input_audio"]["data"]


class TestToNative:
    def test_system_message(self):
        prompt = Prompt(messages=[SystemMessage(content="Be helpful.")])
        result = _to_native_messages(prompt)
        assert result == [{"role": "system", "content": "Be helpful."}]

    def test_user_message(self):
        prompt = Prompt(messages=[UserMessage(content="Hello")])
        result = _to_native_messages(prompt)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_assistant_message_without_tool_calls(self):
        prompt = Prompt(messages=[AssistantMessage(content="Hi there!")])
        result = _to_native_messages(prompt)
        assert result == [{"role": "assistant", "content": "Hi there!"}]

    def test_assistant_message_with_tool_calls(self):
        prompt = Prompt().add_assistant(
            "",
            tool_calls=[
                ToolCall(id="call_1", name="get_weather", arguments={"loc": "Paris"})
            ],
        )
        result = _to_native_messages(prompt)
        assert result[0]["tool_calls"] == [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"loc": "Paris"}'},
            }
        ]

    def test_tool_message(self):
        prompt = Prompt(
            messages=[
                ToolMessage(
                    result=ToolResult(
                        tool_call_id="call_123",
                        name="get_weather",
                        content='{"temp": 72}',
                    )
                )
            ]
        )
        result = _to_native_messages(prompt)
        assert result == [
            {
                "role": "tool",
                "content": '{"temp": 72}',
                "tool_call_id": "call_123",
            }
        ]

    def test_multimodal_user_message(self):
        prompt = Prompt().add_user_content(
            [TextBlock(text="what is this?"), ImageBlock(image="https://x/y.png")]
        )
        result = _to_native_messages(prompt)
        assert result[0]["content"] == [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
        ]

    def test_empty_prompt(self):
        assert _to_native_messages(Prompt(messages=[])) == []


class TestToNativeParams:
    def test_default_config_omits_unset_temperature(self):
        cfg = OpenAIGenerationConfig()
        result = _to_native_params(cfg)
        assert result == {}

    def test_with_fields(self):
        cfg = OpenAIGenerationConfig(
            temperature=0.2,
            max_tokens=100,
            top_p=0.8,
            stop_sequences=["stop"],
            seed=42,
            frequency_penalty=0.3,
            presence_penalty=0.4,
        )
        result = _to_native_params(cfg)
        assert result["temperature"] == 0.2
        assert result["max_tokens"] == 100
        assert result["top_p"] == 0.8
        assert result["stop"] == ["stop"]
        assert result["seed"] == 42
        assert result["frequency_penalty"] == 0.3
        assert result["presence_penalty"] == 0.4

    def test_explicit_temperature_matching_provider_default_is_sent(self):
        cfg = OpenAIGenerationConfig(temperature=0.7)
        result = _to_native_params(cfg)
        assert result["temperature"] == 0.7

    def test_temperature_non_default_non_reasoning_model_is_sent(self):
        cfg = OpenAIGenerationConfig(model="gpt-4.1", temperature=0.2)
        result = _to_native_params(cfg)
        assert result["temperature"] == 0.2

    @pytest.mark.parametrize(
        "model", ["o1", "o1-mini", "o3", "o3-mini", "o4-mini", "gpt-5", "gpt-5-mini"]
    )
    def test_temperature_dropped_for_reasoning_models(self, model):
        cfg = OpenAIGenerationConfig(model=model, temperature=0.2)
        result = _to_native_params(cfg)
        assert "temperature" not in result

    @pytest.mark.parametrize("model", ["o3", "gpt-5", "gpt-6.2-astra"])
    def test_top_p_dropped_with_a_warning_for_reasoning_models(self, model, caplog):
        cfg = OpenAIGenerationConfig(model=model, top_p=0.5, temperature=0.2)
        with caplog.at_level("WARNING"):
            result = _to_native_params(cfg)
        assert "top_p" not in result
        assert "temperature" not in result
        assert "top_p is ignored" in caplog.text
        assert "temperature is ignored" in caplog.text

    @pytest.mark.parametrize("model", ["o4-mini", "gpt-5-mini", "gpt-6.2-sol"])
    def test_reasoning_models_get_max_completion_tokens(self, model):
        cfg = OpenAIGenerationConfig(model=model, max_tokens=100)
        result = _to_native_params(cfg)
        assert result["max_completion_tokens"] == 100
        assert "max_tokens" not in result

    def test_non_reasoning_models_keep_max_tokens_and_top_p(self):
        cfg = OpenAIGenerationConfig(model="gpt-4.1", max_tokens=100, top_p=0.5)
        result = _to_native_params(cfg)
        assert result["max_tokens"] == 100
        assert result["top_p"] == 0.5
        assert "max_completion_tokens" not in result

    def test_temperature_kept_for_gpt5_chat(self):
        cfg = OpenAIGenerationConfig(model="gpt-5-chat", temperature=0.2)
        result = _to_native_params(cfg)
        assert result["temperature"] == 0.2

    def test_reasoning_effort(self):
        cfg = OpenAIGenerationConfig(model="o3", reasoning_effort="high")
        result = _to_native_params(cfg)
        assert result["reasoning_effort"] == "high"

    def test_gpt6_is_a_reasoning_model(self):
        cfg = OpenAIGenerationConfig(
            model="gpt-6.2-astra", temperature=0.2, reasoning_effort="high"
        )
        result = _to_native_params(cfg)
        assert "temperature" not in result
        assert result["reasoning_effort"] == "high"

    def test_gpt6_chat_is_not_a_reasoning_model(self):
        cfg = OpenAIGenerationConfig(model="gpt-6-chat", temperature=0.2)
        assert _to_native_params(cfg)["temperature"] == 0.2

    @pytest.mark.parametrize(
        ("model", "level", "sent"),
        [
            ("gpt-6.2-sol", "none", "low"),
            ("gpt-6.1-sol", "high", "high"),
            ("gpt-6.1-terra", "minimal", "low"),
            ("gpt-6-luna", "none", "none"),
            ("gpt-6-astra", "none", "minimal"),
            ("gpt-6-astra", "xhigh", "xhigh"),
            ("o3", "xhigh", "xhigh"),
            ("gpt-5", "none", "none"),
        ],
    )
    def test_reasoning_effort_is_clamped_to_the_levels_the_model_accepts(
        self, model, level, sent, caplog
    ):
        cfg = OpenAIGenerationConfig(model=model, reasoning_effort=level)
        with caplog.at_level("DEBUG"):
            result = _to_native_params(cfg)
        assert result["reasoning_effort"] == sent
        assert (f"'{level}'" in caplog.text) == (sent != level)

    def test_reasoning_effort_dropped_with_a_warning_for_non_reasoning_model(
        self, caplog
    ):
        cfg = OpenAIGenerationConfig(model="gpt-4.1", reasoning_effort="high")
        with caplog.at_level("WARNING"):
            result = _to_native_params(cfg)
        assert "reasoning_effort" not in result
        assert caplog.text.count("reasoning_effort is ignored") == 1

    def test_reasoning_effort_unset_sends_nothing_and_does_not_warn(self, caplog):
        with caplog.at_level("WARNING"):
            result = _to_native_params(OpenAIGenerationConfig(model="o3"))
        assert "reasoning_effort" not in result
        assert caplog.text == ""

    @pytest.mark.parametrize("value", [True, False])
    def test_parallel_tool_calls_with_tools(self, value):
        cfg = OpenAIGenerationConfig(parallel_tool_calls=value)
        result = _to_native_params(cfg, has_tools=True)
        assert result["parallel_tool_calls"] is value

    def test_parallel_tool_calls_without_tools_is_not_sent(self):
        cfg = OpenAIGenerationConfig(parallel_tool_calls=False)
        assert "parallel_tool_calls" not in _to_native_params(cfg)

    def test_parallel_tool_calls_default_omitted(self):
        cfg = OpenAIGenerationConfig()
        result = _to_native_params(cfg, has_tools=True)
        assert "parallel_tool_calls" not in result

    def test_response_format_json(self):
        cfg = OpenAIGenerationConfig(response_format=ResponseFormat.JSON)
        result = _to_native_params(cfg)
        assert result["response_format"] == {"type": "json_object"}

    def test_response_format_json_schema(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        cfg = OpenAIGenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA, json_schema=schema
        )
        result = _to_native_params(cfg)
        assert result["response_format"] == {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": {**schema, "additionalProperties": False},
                "strict": True,
            },
        }

    def test_response_format_json_schema_not_strict_compatible_sends_strict_false(
        self,
    ):
        class _Defaulted(BaseModel):
            name: str
            note: str = "n/a"

        schema = model_schema(_Defaulted)
        cfg = OpenAIGenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA, json_schema=schema
        )

        result = _to_native_params(cfg)

        assert result["response_format"]["json_schema"]["strict"] is False
        assert result["response_format"]["json_schema"]["schema"] == schema

    def test_response_format_json_schema_name_comes_from_the_title(self):
        schema = {
            "title": "My Label!",
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        cfg = OpenAIGenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA, json_schema=schema
        )
        result = _to_native_params(cfg)
        assert result["response_format"]["json_schema"]["name"] == "MyLabel"

    def test_json_schema_missing_raises(self):
        cfg = OpenAIGenerationConfig(
            response_format=ResponseFormat.JSON_SCHEMA, json_schema=None
        )
        with pytest.raises(ValueError, match="json_schema is required"):
            _to_native_params(cfg)

    def test_extra_params_applied(self):
        cfg = OpenAIGenerationConfig(extra_params={"user": "u1"})
        result = _to_native_params(cfg)
        assert result["user"] == "u1"


class TestStrictJsonSchema:
    def test_walks_arrays_and_unions(self):
        class _Item(BaseModel):
            name: str

        class _Model(BaseModel):
            items: list[_Item]
            other: _Item | None

        result = strict_json_schema(model_schema(_Model))

        assert result is not None
        assert result["$defs"]["_Item"]["additionalProperties"] is False

    def test_object_without_properties_below_the_top_is_not_strict(self):
        schema = {
            "type": "object",
            "properties": {"data": {"type": "object"}},
            "required": ["data"],
        }
        assert strict_json_schema(schema) is None

    def test_untyped_property_is_not_strict(self):
        schema = {
            "type": "object",
            "properties": {"data": {"title": "Data"}},
            "required": ["data"],
        }
        assert strict_json_schema(schema) is None

    def test_array_of_objects_without_properties_is_not_strict(self):
        schema = {
            "type": "object",
            "properties": {"rows": {"type": "array", "items": {"type": "object"}}},
            "required": ["rows"],
        }
        assert strict_json_schema(schema) is None

    @pytest.mark.parametrize(
        "array_schema",
        [{"type": "array"}, {"type": "array", "items": {}}],
        ids=["missing-items", "empty-items"],
    )
    def test_array_with_untyped_items_is_not_strict(self, array_schema):
        schema = {
            "type": "object",
            "properties": {"tags": array_schema},
            "required": ["tags"],
        }
        assert strict_json_schema(schema) is None

    def test_required_bare_list_field_is_not_strict(self):
        class _Model(BaseModel):
            tags: list

        assert strict_json_schema(model_schema(_Model)) is None

    def test_explicit_additional_properties_true_is_not_strict(self):
        schema = {"type": "object", "properties": {}, "additionalProperties": True}
        assert strict_json_schema(schema) is None

    def test_the_input_is_left_untouched(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
        }
        before = copy.deepcopy(schema)

        strict_json_schema(schema)

        assert schema == before


class TestFromNative:
    def test_simple_message(self):
        response = make_chat_completion(
            "Hello world", prompt_tokens=1, completion_tokens=2
        )
        result = _from_native_response(response, model="gpt-4.1")
        assert result.message.content == "Hello world"
        assert result.message.tool_calls == []
        assert result.model == "gpt-4.1"
        assert result.finish_reason == FinishReason.STOP
        assert result.usage.input_tokens == 1
        assert result.usage.output_tokens == 2

    def test_with_tool_calls(self):
        response = make_chat_completion(
            None,
            tool_calls=[
                make_chat_tool_call("call_abc", "get_weather", '{"location": "Paris"}')
            ],
        )
        result = _from_native_response(response, model="gpt-4.1")
        assert result.message.content == ""
        tc = result.message.tool_calls[0]
        assert tc.id == "call_abc"
        assert tc.name == "get_weather"
        assert tc.arguments == {"location": "Paris"}

    def test_no_tool_calls(self):
        response = make_chat_completion("Hi")
        result = _from_native_response(response, model="gpt-4.1")
        assert result.message.tool_calls == []
